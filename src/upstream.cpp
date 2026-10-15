#include "openai_api/upstream.hpp"
#include "openai_api/encoder/encoder.hpp"

#include <ctime>
#include <random>
#include <regex>
#include <stdexcept>

namespace openai_api {

using Clock = std::chrono::steady_clock;

namespace {

double ms_between(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

struct ParsedUrl {
    std::string host_port;    // scheme://host:port
    std::string path_prefix;  // 不带结尾 '/'
};

ParsedUrl parse_base_url(const std::string& url) {
    static const std::regex re(R"(^(https?)://([^/\s]+)(/[^\s]*)?$)");
    std::smatch m;
    if (!std::regex_match(url, m, re)) {
        throw std::invalid_argument("invalid base_url: " + url);
    }
#ifndef CPPHTTPLIB_OPENSSL_SUPPORT
    if (m[1].str() == "https") {
        throw std::invalid_argument("https upstream requires CPPHTTPLIB_OPENSSL_SUPPORT: " + url);
    }
#endif
    ParsedUrl out;
    out.host_port = m[1].str() + "://" + m[2].str();
    out.path_prefix = m[3].matched ? m[3].str() : "";
    while (!out.path_prefix.empty() && out.path_prefix.back() == '/') {
        out.path_prefix.pop_back();
    }
    return out;
}

std::string make_boundary() {
    static const char hex[] = "0123456789abcdef";
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 15);
    std::string b = "----openai-api-upstream-";
    for (int i = 0; i < 24; ++i) b += hex[dist(gen)];
    return b;
}

// httplib 解析 multipart 后不保留原始 body，这里按解析结果重新编码
std::string encode_multipart(const httplib::MultipartFormData& form, const std::string& boundary,
                             const std::string& model_override) {
    std::string out;
    for (const auto& kv : form.fields) {
        const auto& f = kv.second;
        out += "--" + boundary + "\r\n";
        out += "Content-Disposition: form-data; name=\"" + f.name + "\"\r\n\r\n";
        out += (f.name == "model" && !model_override.empty()) ? model_override : f.content;
        out += "\r\n";
    }
    for (const auto& kv : form.files) {
        const auto& f = kv.second;
        out += "--" + boundary + "\r\n";
        out += "Content-Disposition: form-data; name=\"" + f.name + "\"; filename=\"" + f.filename + "\"\r\n";
        out += "Content-Type: " + (f.content_type.empty() ? std::string("application/octet-stream") : f.content_type) + "\r\n\r\n";
        out += f.content;
        out += "\r\n";
    }
    out += "--" + boundary + "--\r\n";
    return out;
}

bool starts_with(const std::string& s, const char* prefix) {
    return s.rfind(prefix, 0) == 0;
}

// 从 SSE 流中提取首个 token 时间和 usage，不修改数据
class SSEStatsTracker {
public:
    void feed(const char* data, size_t n, Clock::time_point now) {
        buf_.append(data, n);
        size_t pos;
        while ((pos = buf_.find('\n')) != std::string::npos) {
            std::string line = buf_.substr(0, pos);
            buf_.erase(0, pos + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            onLine(line, now);
        }
        if (buf_.size() > (1u << 20)) buf_.clear();  // 异常的超长行，放弃统计
    }

    std::optional<Clock::time_point> first_token;
    nlohmann::json usage;

private:
    void onLine(const std::string& line, Clock::time_point now) {
        if (!starts_with(line, "data:")) return;
        auto payload = line.substr(5);
        payload.erase(0, payload.find_first_not_of(" \t"));
        if (payload.empty() || payload == "[DONE]") return;
        auto j = nlohmann::json::parse(payload, nullptr, false);
        if (j.is_discarded() || !j.is_object()) return;
        if (!first_token && hasToken(j)) first_token = now;
        if (j.contains("usage") && j["usage"].is_object()) usage = j["usage"];
    }

    static bool hasToken(const nlohmann::json& j) {
        if (!j.contains("choices") || !j["choices"].is_array()) return false;
        for (const auto& c : j["choices"]) {
            if (!c.is_object()) continue;
            if (c.contains("delta") && c["delta"].is_object()) {
                const auto& d = c["delta"];
                for (const char* key : {"content", "reasoning_content"}) {
                    if (d.contains(key) && d[key].is_string() && !d[key].get<std::string>().empty()) return true;
                }
                if (d.contains("tool_calls") && !d["tool_calls"].is_null()) return true;
            }
            if (c.contains("text") && c["text"].is_string() && !c["text"].get<std::string>().empty()) return true;
        }
        return false;
    }

    std::string buf_;
};

// 一次上游调用的共享状态（上游线程写，响应线程读）
struct ProxyCall {
    std::mutex m;
    std::condition_variable cv;
    bool headers_ready = false;
    bool finished = false;
    std::atomic<bool> cancelled{false};
    int status = 0;
    std::string content_type;
    std::string pending;     // 尚未转给客户端的数据
    std::string error;       // 传输错误（没连上或中途断开）
};

std::optional<double> usage_number(const nlohmann::json& usage, const char* key) {
    if (usage.is_object() && usage.contains(key) && usage[key].is_number()) return usage[key].get<double>();
    return std::nullopt;
}

// 优先使用上游自己报告的解码速度（如 axllm 的 usage.decode_tps），否则按 completion_tokens / 生成耗时估算
std::optional<double> tokens_per_second(const nlohmann::json& usage, double gen_ms) {
    if (auto tps = usage_number(usage, "decode_tps"); tps && *tps > 0) return tps;
    if (!usage.is_object() || !usage.contains("completion_tokens") || !usage["completion_tokens"].is_number()) {
        return std::nullopt;
    }
    double tokens = usage["completion_tokens"].get<double>();
    if (tokens <= 0 || gen_ms <= 0) return std::nullopt;
    return tokens * 1000.0 / gen_ms;
}

} // namespace

// ============ 序列化 ============

nlohmann::json UpstreamOptions::to_json() const {
    nlohmann::json j;
    j["base_url"] = base_url;
    j["upstream_model"] = upstream_model;
    j["has_api_key"] = !api_key.empty();
    j["health_path"] = health_path;
    j["connect_timeout_ms"] = connect_timeout.count();
    j["read_timeout_ms"] = read_timeout.count();
    j["extra"] = extra.is_object() ? extra : nlohmann::json::object();
    return j;
}

UpstreamOptions UpstreamOptions::from_json(const nlohmann::json& j) {
    UpstreamOptions o;
    o.base_url = j.value("base_url", std::string());
    o.upstream_model = j.value("upstream_model", std::string());
    o.api_key = j.value("api_key", std::string());
    if (j.contains("health_path") && j["health_path"].is_string()) o.health_path = j["health_path"];
    if (j.contains("connect_timeout_ms")) o.connect_timeout = std::chrono::milliseconds(j["connect_timeout_ms"].get<int64_t>());
    if (j.contains("read_timeout_ms")) o.read_timeout = std::chrono::milliseconds(j["read_timeout_ms"].get<int64_t>());
    if (j.contains("extra") && j["extra"].is_object()) o.extra = j["extra"];
    return o;
}

const char* to_string(UpstreamHealth health) {
    switch (health) {
        case UpstreamHealth::Healthy: return "healthy";
        case UpstreamHealth::Unhealthy: return "unhealthy";
        default: return "unknown";
    }
}

nlohmann::json UpstreamCallStats::to_json() const {
    nlohmann::json j;
    j["endpoint"] = endpoint;
    j["status"] = status;
    j["stream"] = stream;
    j["started_at"] = started_at;
    j["duration_ms"] = duration_ms;
    j["ttft_ms"] = ttft_ms ? nlohmann::json(*ttft_ms) : nlohmann::json(nullptr);
    j["usage"] = usage;
    j["tokens_per_second"] = tokens_per_second ? nlohmann::json(*tokens_per_second) : nlohmann::json(nullptr);
    return j;
}

nlohmann::json UpstreamInfo::to_json() const {
    nlohmann::json j = options.to_json();
    j["model"] = model;
    j["health"] = to_string(health);
    j["model_info"] = model_info.is_object() ? model_info : nlohmann::json::object();
    j["requests"] = requests;
    j["errors"] = errors;
    j["active"] = active;
    j["last_error"] = last_error;
    j["last_call"] = last_call ? last_call->to_json() : nlohmann::json(nullptr);
    return j;
}

// ============ 注册表 ============

struct UpstreamManager::Entry {
    std::string model;
    UpstreamOptions options;
    ParsedUrl url;

    std::mutex mutex;
    UpstreamHealth health = UpstreamHealth::Unknown;
    int consecutive_failures = 0;
    nlohmann::json model_info = nlohmann::json::object();
    uint64_t requests = 0;
    uint64_t errors = 0;
    int active = 0;
    std::string last_error;
    std::optional<UpstreamCallStats> last_call;

    // 调用结束：更新计数和最近一次调用
    void finish(UpstreamCallStats stats, const std::string& error) {
        std::lock_guard<std::mutex> lock(mutex);
        active--;
        if (stats.status == 0 || stats.status >= 400) {
            errors++;
            last_error = !error.empty() ? error : "HTTP " + std::to_string(stats.status);
        }
        // 连不上上游：被动标记，由健康检查恢复
        if (stats.status == 0 && !options.health_path.empty() && ++consecutive_failures >= 2) {
            health = UpstreamHealth::Unhealthy;
        }
        last_call = std::move(stats);
    }
};

UpstreamManager::UpstreamManager() = default;

UpstreamManager::~UpstreamManager() {
    stopHealthCheck();
}

void UpstreamManager::add(const std::string& model_name, UpstreamOptions options) {
    if (model_name.empty()) throw std::invalid_argument("model name is empty");
    auto entry = std::make_shared<Entry>();
    entry->model = model_name;
    entry->url = parse_base_url(options.base_url);
    entry->options = std::move(options);
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        entries_[model_name] = entry;
    }
    {
        std::lock_guard<std::mutex> lock(health_mutex_);
        health_wakeup_ = true;
    }
    health_cv_.notify_all();
    notifyChange();
}

bool UpstreamManager::remove(const std::string& model_name) {
    bool removed;
    {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        removed = entries_.erase(model_name) > 0;
    }
    if (removed) notifyChange();
    return removed;
}

bool UpstreamManager::has(const std::string& model_name) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return entries_.count(model_name) > 0;
}

bool UpstreamManager::empty() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    return entries_.empty();
}

std::vector<std::string> UpstreamManager::names() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::vector<std::string> out;
    for (const auto& kv : entries_) out.push_back(kv.first);
    return out;
}

std::vector<UpstreamInfo> UpstreamManager::list() const {
    std::vector<std::shared_ptr<Entry>> snapshot;
    {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        for (const auto& kv : entries_) snapshot.push_back(kv.second);
    }
    std::vector<UpstreamInfo> out;
    for (const auto& e : snapshot) {
        std::lock_guard<std::mutex> lock(e->mutex);
        UpstreamInfo info;
        info.model = e->model;
        info.options = e->options;
        info.health = e->health;
        info.model_info = e->model_info;
        info.requests = e->requests;
        info.errors = e->errors;
        info.active = e->active;
        info.last_error = e->last_error;
        info.last_call = e->last_call;
        out.push_back(std::move(info));
    }
    return out;
}

std::optional<nlohmann::json> UpstreamManager::listedModelFields(const std::string& model_name) const {
    auto e = find(model_name);
    if (!e) return std::nullopt;
    std::lock_guard<std::mutex> lock(e->mutex);
    if (e->health == UpstreamHealth::Unhealthy) return std::nullopt;
    nlohmann::json fields = e->model_info.is_object() ? e->model_info : nlohmann::json::object();
    if (e->options.extra.is_object()) {
        for (auto it = e->options.extra.begin(); it != e->options.extra.end(); ++it) {
            fields[it.key()] = it.value();
        }
    }
    return fields;
}

std::shared_ptr<UpstreamManager::Entry> UpstreamManager::find(const std::string& model_name) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = entries_.find(model_name);
    return it == entries_.end() ? nullptr : it->second;
}

void UpstreamManager::setOnChange(std::function<void()> callback) {
    std::lock_guard<std::mutex> lock(change_mutex_);
    on_change_ = std::move(callback);
}

void UpstreamManager::notifyChange() {
    std::function<void()> cb;
    {
        std::lock_guard<std::mutex> lock(change_mutex_);
        cb = on_change_;
    }
    if (cb) cb();
}

// ============ 透传 ============

bool UpstreamManager::tryProxy(const httplib::Request& req, httplib::Response& res) {
    if (empty()) return false;

    // 取出请求中的 model
    std::string model;
    nlohmann::json body_json;
    const bool multipart = req.is_multipart_form_data();
    if (multipart) {
        if (req.form.has_field("model")) model = req.form.get_field("model");
    } else {
        body_json = nlohmann::json::parse(req.body, nullptr, false);
        if (body_json.is_object() && body_json.contains("model") && body_json["model"].is_string()) {
            model = body_json["model"].get<std::string>();
        }
    }
    if (model.empty()) return false;

    auto entry = find(model);
    if (!entry) return false;

    {
        std::lock_guard<std::mutex> lock(entry->mutex);
        if (entry->health == UpstreamHealth::Unhealthy) {
            res.status = 503;
            res.set_content(ErrorEncoder::encode("model_not_ready", "Model '" + model + "' is not ready", 503),
                            "application/json");
            return true;
        }
        entry->requests++;
        entry->active++;
    }

    const UpstreamOptions& opt = entry->options;

    // 组装转发请求：除了可选的模型改名，body 原样转发
    httplib::Request ureq;
    ureq.method = req.method;
    ureq.path = entry->url.path_prefix + (req.target.empty() ? req.path : req.target);
    std::string content_type;
    if (multipart) {
        auto boundary = make_boundary();
        ureq.body = encode_multipart(req.form, boundary, opt.upstream_model);
        content_type = "multipart/form-data; boundary=" + boundary;
    } else {
        if (!opt.upstream_model.empty() && body_json.is_object()) {
            body_json["model"] = opt.upstream_model;
            ureq.body = body_json.dump();
        } else {
            ureq.body = req.body;
        }
        content_type = req.get_header_value("Content-Type");
        if (content_type.empty()) content_type = "application/json";
    }
    ureq.set_header("Content-Type", content_type);
    if (req.has_header("Accept")) ureq.set_header("Accept", req.get_header_value("Accept"));
    if (!opt.api_key.empty()) ureq.set_header("Authorization", "Bearer " + opt.api_key);

    UpstreamCallStats stats;
    stats.endpoint = req.path;
    stats.stream = body_json.is_object() && body_json.value("stream", false);
    stats.started_at = static_cast<int64_t>(std::time(nullptr));
    const auto start = Clock::now();

    // 上游请求在独立线程里执行，数据通过 ProxyCall 交给响应线程
    auto call = std::make_shared<ProxyCall>();
    ureq.response_handler = [call](const httplib::Response& r) {
        std::lock_guard<std::mutex> lock(call->m);
        call->status = r.status;
        call->content_type = r.get_header_value("Content-Type");
        call->headers_ready = true;
        call->cv.notify_all();
        return !call->cancelled.load();
    };
    ureq.content_receiver = [call](const char* data, size_t n, uint64_t, uint64_t) {
        if (call->cancelled.load()) return false;
        std::lock_guard<std::mutex> lock(call->m);
        call->pending.append(data, n);
        call->cv.notify_all();
        return true;
    };
    std::thread([call, ureq = std::move(ureq), host = entry->url.host_port,
                 ct = opt.connect_timeout, rt = opt.read_timeout]() mutable {
        httplib::Client cli(host);
        cli.set_connection_timeout(ct);
        cli.set_read_timeout(rt);
        cli.set_write_timeout(rt);
        httplib::Response ures;
        httplib::Error err = httplib::Error::Success;
        bool ok = cli.send(ureq, ures, err);
        std::lock_guard<std::mutex> lock(call->m);
        if (!ok && !call->cancelled.load()) call->error = httplib::to_string(err);
        call->finished = true;
        call->cv.notify_all();
    }).detach();

    // 等上游返回响应头
    {
        std::unique_lock<std::mutex> lock(call->m);
        call->cv.wait_for(lock, opt.read_timeout, [&] { return call->headers_ready || call->finished; });
        if (!call->headers_ready) {
            std::string error = call->error.empty() ? "timeout waiting for upstream response" : call->error;
            call->cancelled = true;
            lock.unlock();
            res.status = 502;
            res.set_content(ErrorEncoder::encode("upstream_unavailable",
                                                 "Upstream for model '" + model + "' is unavailable: " + error, 502),
                            "application/json");
            stats.duration_ms = ms_between(start, Clock::now());
            entry->finish(std::move(stats), error);
            return true;
        }
        stats.status = call->status;
        res.status = call->status;
    }

    if (starts_with(call->content_type, "text/event-stream")) {
        // 流式：逐块原样转发
        struct StreamCtx {
            std::shared_ptr<ProxyCall> call;
            std::shared_ptr<Entry> entry;
            UpstreamCallStats stats;
            Clock::time_point start;
            SSEStatsTracker tracker;
            std::atomic<bool> finalized{false};
        };
        auto ctx = std::make_shared<StreamCtx>();
        ctx->call = call;
        ctx->entry = entry;
        ctx->stats = std::move(stats);
        ctx->start = start;

        res.set_header("Cache-Control", "no-cache");
        res.set_chunked_content_provider(
            call->content_type,
            [ctx](size_t, httplib::DataSink& sink) {
                std::string data;
                bool finished;
                {
                    std::unique_lock<std::mutex> lock(ctx->call->m);
                    ctx->call->cv.wait_for(lock, std::chrono::milliseconds(100), [&] {
                        return !ctx->call->pending.empty() || ctx->call->finished;
                    });
                    data.swap(ctx->call->pending);
                    finished = ctx->call->finished;
                }
                if (!data.empty()) {
                    ctx->tracker.feed(data.data(), data.size(), Clock::now());
                    if (!sink.write(data.data(), data.size())) {
                        ctx->call->cancelled = true;
                        return false;
                    }
                }
                if (finished && data.empty()) {
                    sink.done();
                }
                return true;
            },
            [ctx](bool success) {
                if (!success) ctx->call->cancelled = true;
                if (ctx->finalized.exchange(true)) return;
                auto end = Clock::now();
                auto& s = ctx->stats;
                s.duration_ms = ms_between(ctx->start, end);
                if (ctx->tracker.first_token) {
                    s.ttft_ms = ms_between(ctx->start, *ctx->tracker.first_token);
                    s.tokens_per_second = tokens_per_second(ctx->tracker.usage, ms_between(*ctx->tracker.first_token, end));
                }
                s.usage = ctx->tracker.usage;
                std::string error;
                {
                    std::lock_guard<std::mutex> lock(ctx->call->m);
                    error = ctx->call->error;
                }
                if (!success && error.empty()) error = "client disconnected";
                ctx->entry->finish(std::move(s), error);
            });
        return true;
    }

    // 非流式：等上游完成后整体返回
    std::string body, error;
    {
        std::unique_lock<std::mutex> lock(call->m);
        bool done = call->cv.wait_for(lock, opt.read_timeout, [&] { return call->finished; });
        if (!done) call->cancelled = true;
        body.swap(call->pending);
        error = done ? call->error : "timeout waiting for upstream response";
    }
    stats.duration_ms = ms_between(start, Clock::now());
    if (!error.empty()) {
        res.status = 502;
        res.set_content(ErrorEncoder::encode("upstream_unavailable",
                                             "Upstream for model '" + model + "' failed: " + error, 502),
                        "application/json");
        stats.status = 0;
        entry->finish(std::move(stats), error);
        return true;
    }
    if (starts_with(call->content_type, "application/json")) {
        auto j = nlohmann::json::parse(body, nullptr, false);
        if (j.is_object() && j.contains("usage") && j["usage"].is_object()) {
            stats.usage = j["usage"];
            stats.tokens_per_second = tokens_per_second(stats.usage, stats.duration_ms);
            stats.ttft_ms = usage_number(stats.usage, "ttft_ms");  // 非流式只能用上游报告的首字延迟
        }
    }
    res.set_content(std::move(body), call->content_type.empty() ? "application/octet-stream" : call->content_type);
    entry->finish(std::move(stats), "");
    return true;
}

// ============ 健康检查 ============

void UpstreamManager::startHealthCheck(std::chrono::milliseconds interval) {
    if (interval.count() <= 0) return;
    std::lock_guard<std::mutex> lock(health_mutex_);
    if (health_thread_.joinable()) return;
    health_interval_ = interval;
    health_stop_ = false;
    health_thread_ = std::thread([this] { healthLoop(); });
}

void UpstreamManager::stopHealthCheck() {
    std::thread t;
    {
        std::lock_guard<std::mutex> lock(health_mutex_);
        health_stop_ = true;
        t = std::move(health_thread_);
    }
    health_cv_.notify_all();
    if (t.joinable()) t.join();
}

void UpstreamManager::healthLoop() {
    std::unique_lock<std::mutex> lock(health_mutex_);
    while (!health_stop_) {
        health_wakeup_ = false;
        lock.unlock();
        std::vector<std::shared_ptr<Entry>> snapshot;
        {
            std::shared_lock<std::shared_mutex> l(mutex_);
            for (const auto& kv : entries_) snapshot.push_back(kv.second);
        }
        for (const auto& e : snapshot) checkHealth(e);
        lock.lock();
        health_cv_.wait_for(lock, health_interval_, [this] { return health_stop_ || health_wakeup_; });
    }
}

void UpstreamManager::checkHealth(const std::shared_ptr<Entry>& entry) {
    const auto& opt = entry->options;
    if (opt.health_path.empty()) return;

    httplib::Client cli(entry->url.host_port);
    cli.set_connection_timeout(std::chrono::milliseconds(2000));
    cli.set_read_timeout(std::chrono::milliseconds(3000));
    httplib::Headers headers;
    if (!opt.api_key.empty()) headers.emplace("Authorization", "Bearer " + opt.api_key);
    auto r = cli.Get(entry->url.path_prefix + opt.health_path, headers);
    const bool ok = r && r->status == 200;

    // 健康检查路径是模型列表时，顺便取出该模型的字段（如上下文长度）
    nlohmann::json info;
    if (ok) {
        auto j = nlohmann::json::parse(r->body, nullptr, false);
        if (j.is_object() && j.contains("data") && j["data"].is_array()) {
            const std::string want = opt.upstream_model.empty() ? entry->model : opt.upstream_model;
            const nlohmann::json* match = nullptr;
            for (const auto& m : j["data"]) {
                if (m.is_object() && m.value("id", std::string()) == want) match = &m;
            }
            if (!match && j["data"].size() == 1 && j["data"][0].is_object()) match = &j["data"][0];
            if (match) {
                info = nlohmann::json::object();
                for (auto it = match->begin(); it != match->end(); ++it) {
                    if (it.key() == "id" || it.key() == "object" || it.key() == "created" || it.key() == "owned_by") {
                        continue;
                    }
                    info[it.key()] = it.value();
                }
            }
        }
    }

    std::lock_guard<std::mutex> lock(entry->mutex);
    if (ok) {
        entry->health = UpstreamHealth::Healthy;
        entry->consecutive_failures = 0;
        if (info.is_object()) entry->model_info = std::move(info);
    } else {
        entry->last_error = r ? "health check HTTP " + std::to_string(r->status)
                              : "health check failed: " + httplib::to_string(r.error());
        if (++entry->consecutive_failures >= 2) entry->health = UpstreamHealth::Unhealthy;
    }
}

} // namespace openai_api
