#include "openai_api/server.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

using namespace openai_api;

namespace {

const int BACKEND_PORT = 18301;
const int GATEWAY_PORT = 18302;
const int DEAD_PORT = 18309;  // 没有服务监听

const char* SSE_BODY =
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"}}]}\n\n"
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hel\"}}]}\n\n"
    "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"}}]}\n\n"
    "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":2,\"total_tokens\":14}}\n\n"
    "data: [DONE]\n\n";

// 模拟上游：一个普通的 OpenAI 兼容服务
struct Backend {
    httplib::Server http;
    std::thread thread;
    std::atomic<bool> stream_aborted{false};
    std::string last_auth;
    std::string last_model;

    void start() {
        http.Get("/v1/models", [](const httplib::Request&, httplib::Response& res) {
            res.set_content(R"({"object":"list","data":[{"id":"backend-model","object":"model","owned_by":"x","max_token_len":4096}]})",
                            "application/json");
        });
        http.Post("/v1/chat/completions", [this](const httplib::Request& req, httplib::Response& res) {
            auto j = nlohmann::json::parse(req.body);
            last_auth = req.get_header_value("Authorization");
            last_model = j.value("model", "");
            const std::string mode = j.value("mode", "");
            if (mode == "error") {
                res.status = 400;
                res.set_content(R"({"error":{"message":"bad","type":"invalid_request_error"}})", "application/json");
                return;
            }
            if (mode == "slow") {
                res.set_chunked_content_provider("text/event-stream", [this](size_t offset, httplib::DataSink& sink) {
                    if (offset > 100000) {
                        sink.done();
                        return true;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(20));
                    std::string chunk = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"x\"}}]}\n\n";
                    if (!sink.write(chunk.data(), chunk.size())) {
                        stream_aborted = true;
                        return false;
                    }
                    return true;
                }, [this](bool success) {
                    if (!success) stream_aborted = true;
                });
                return;
            }
            if (j.value("stream", false)) {
                res.set_chunked_content_provider("text/event-stream", [](size_t, httplib::DataSink& sink) {
                    std::string body = SSE_BODY;
                    // 分成小块写，验证网关逐块转发
                    for (size_t i = 0; i < body.size(); i += 17) {
                        auto part = body.substr(i, 17);
                        sink.write(part.data(), part.size());
                        std::this_thread::sleep_for(std::chrono::milliseconds(2));
                    }
                    sink.done();
                    return true;
                });
                return;
            }
            nlohmann::json out;
            out["id"] = "chatcmpl-1";
            out["object"] = "chat.completion";
            out["model"] = last_model;
            out["choices"] = nlohmann::json::array({{{"index", 0},
                {"message", {{"role", "assistant"}, {"content", "Hello"}, {"reasoning_content", "think"}}},
                {"finish_reason", "stop"}}});
            out["usage"] = {{"prompt_tokens", 10}, {"completion_tokens", 5}, {"total_tokens", 15}};
            res.set_content(out.dump(), "application/json");
        });
        http.Post("/v1/audio/transcriptions", [](const httplib::Request& req, httplib::Response& res) {
            nlohmann::json out;
            out["model"] = req.form.get_field("model");
            out["language"] = req.form.get_field("language");
            out["file"] = req.form.has_file("file") ? req.form.get_file("file").content : "";
            out["filename"] = req.form.has_file("file") ? req.form.get_file("file").filename : "";
            res.set_content(out.dump(), "application/json");
        });
        thread = std::thread([this] { http.listen("127.0.0.1", BACKEND_PORT); });
        http.wait_until_ready();
    }

    void stop() {
        http.stop();
        if (thread.joinable()) thread.join();
    }
};

httplib::Client make_client() {
    httplib::Client c("127.0.0.1", GATEWAY_PORT);
    c.set_connection_timeout(std::chrono::seconds(3));
    c.set_read_timeout(std::chrono::seconds(10));
    return c;
}

std::optional<UpstreamInfo> find_upstream(Server& server, const std::string& model) {
    for (auto& info : server.listUpstreams()) {
        if (info.model == model) return info;
    }
    return std::nullopt;
}

template <typename F>
bool wait_for(F cond, int timeout_ms = 3000) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (cond()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return cond();
}

void test_non_stream_passthrough(Server& server) {
    std::cout << "Test: non_stream_passthrough... " << std::flush;
    auto c = make_client();
    auto r = c.Post("/v1/chat/completions",
                    R"({"model":"qwen","messages":[{"role":"user","content":"hi"}]})", "application/json");
    assert(r && r->status == 200);
    auto j = nlohmann::json::parse(r->body);
    // 未改写：上游自定义字段原样返回
    assert(j["choices"][0]["message"]["reasoning_content"] == "think");
    assert(j["model"] == "qwen");

    auto info = find_upstream(server, "qwen");
    assert(info && info->last_call);
    assert(info->last_call->usage["total_tokens"] == 15);
    assert(!info->last_call->stream);
    std::cout << "PASSED" << std::endl;
}

void test_stream_passthrough(Server& server) {
    std::cout << "Test: stream_passthrough... " << std::flush;
    auto c = make_client();
    std::string received;
    int chunks = 0;
    httplib::Request req;
    req.method = "POST";
    req.path = "/v1/chat/completions";
    req.body = R"({"model":"qwen","stream":true,"messages":[{"role":"user","content":"hi"}]})";
    req.set_header("Content-Type", "application/json");
    req.content_receiver = [&](const char* d, size_t n, uint64_t, uint64_t) {
        received.append(d, n);
        chunks++;
        return true;
    };
    auto r = c.send(req);
    assert(r && r->status == 200);
    assert(r->get_header_value("Content-Type").find("text/event-stream") == 0);
    assert(received == SSE_BODY);  // 字节级一致
    assert(chunks > 1);             // 逐块转发，不是一次性返回

    auto info = find_upstream(server, "qwen");
    assert(wait_for([&] {
        info = find_upstream(server, "qwen");
        return info && info->last_call && info->last_call->stream;
    }));
    assert(info->last_call->usage["prompt_tokens"] == 12);
    assert(info->last_call->ttft_ms.has_value());
    assert(info->active == 0);
    std::cout << "PASSED" << std::endl;
}

void test_model_rename_and_upstream_key(Server& server, Backend& backend) {
    std::cout << "Test: model_rename_and_upstream_key... " << std::flush;
    UpstreamOptions up;
    up.base_url = "http://127.0.0.1:" + std::to_string(BACKEND_PORT);
    up.upstream_model = "backend-model";
    up.api_key = "sk-upstream";
    server.registerUpstream("renamed", up);

    auto c = make_client();
    auto r = c.Post("/v1/chat/completions", R"({"model":"renamed","messages":[]})", "application/json");
    assert(r && r->status == 200);
    assert(backend.last_model == "backend-model");
    assert(backend.last_auth == "Bearer sk-upstream");
    server.unregisterUpstream("renamed");
    std::cout << "PASSED" << std::endl;
}

void test_multipart_passthrough() {
    std::cout << "Test: multipart_passthrough... " << std::flush;
    auto c = make_client();
    httplib::UploadFormDataItems items = {
        {"model", "qwen", "", ""},
        {"language", "zh", "", ""},
        {"file", std::string("RIFF\0\x01audio", 11), "a.wav", "audio/wav"},
    };
    auto r = c.Post("/v1/audio/transcriptions", items);
    assert(r && r->status == 200);
    auto j = nlohmann::json::parse(r->body);
    assert(j["model"] == "qwen");
    assert(j["language"] == "zh");
    assert(j["filename"] == "a.wav");
    assert(j["file"].get<std::string>() == std::string("RIFF\0\x01audio", 11));
    std::cout << "PASSED" << std::endl;
}

void test_upstream_error_passthrough() {
    std::cout << "Test: upstream_error_passthrough... " << std::flush;
    auto c = make_client();
    auto r = c.Post("/v1/chat/completions", R"({"model":"qwen","mode":"error"})", "application/json");
    assert(r && r->status == 400);
    assert(nlohmann::json::parse(r->body)["error"]["message"] == "bad");
    std::cout << "PASSED" << std::endl;
}

void test_models_listing(Server& server) {
    std::cout << "Test: models_listing... " << std::flush;
    // 健康检查读取上游 /v1/models 中的字段
    assert(wait_for([&] {
        auto info = find_upstream(server, "qwen");
        return info && info->health == UpstreamHealth::Healthy && info->model_info.contains("max_token_len");
    }));
    auto c = make_client();
    auto r = c.Get("/v1/models");
    assert(r && r->status == 200);
    auto j = nlohmann::json::parse(r->body);
    bool found = false;
    for (const auto& m : j["data"]) {
        if (m["id"] == "qwen") {
            found = true;
            assert(m["max_token_len"] == 4096);
            assert(m["context"] == "extra");
        }
    }
    assert(found);
    std::cout << "PASSED" << std::endl;
}

void test_admin_api(Server& server) {
    std::cout << "Test: admin_api... " << std::flush;
    auto c = make_client();
    auto r = c.Post("/admin/upstreams",
                    R"({"model":"org/m-1","base_url":"http://127.0.0.1:)" + std::to_string(BACKEND_PORT) + R"("})",
                    "application/json");
    assert(r && r->status == 200);
    assert(server.hasModel("org/m-1"));

    r = c.Get("/admin/upstreams");
    assert(r && r->status == 200);
    auto j = nlohmann::json::parse(r->body);
    assert(j["data"].size() == 2);
    for (const auto& u : j["data"]) assert(!u.contains("api_key"));  // 不回显 key

    r = c.Post("/admin/upstreams", R"({"model":"bad","base_url":"not a url"})", "application/json");
    assert(r && r->status == 400);

    r = c.Delete("/admin/upstreams/org%2Fm-1");
    assert(r && r->status == 200);
    assert(!server.hasModel("org/m-1"));
    r = c.Delete("/admin/upstreams/org%2Fm-1");
    assert(r && r->status == 404);
    std::cout << "PASSED" << std::endl;
}

void test_unknown_model_falls_through() {
    std::cout << "Test: unknown_model_falls_through... " << std::flush;
    auto c = make_client();
    auto r = c.Post("/v1/chat/completions", R"({"model":"nope","messages":[]})", "application/json");
    assert(r && r->status == 400);  // 走原有的本地路由逻辑
    r = c.Post("/v1/completions", R"({"model":"nope","prompt":"x"})", "application/json");
    assert(r && r->status == 404);
    std::cout << "PASSED" << std::endl;
}

void test_client_disconnect_cancels_upstream(Backend& backend) {
    std::cout << "Test: client_disconnect_cancels_upstream... " << std::flush;
    auto c = make_client();
    int chunks = 0;
    httplib::Request req;
    req.method = "POST";
    req.path = "/v1/chat/completions";
    req.body = R"({"model":"qwen","stream":true,"mode":"slow"})";
    req.set_header("Content-Type", "application/json");
    req.content_receiver = [&](const char*, size_t, uint64_t, uint64_t) { return ++chunks < 3; };
    c.send(req);
    assert(wait_for([&] { return backend.stream_aborted.load(); }, 5000));
    std::cout << "PASSED" << std::endl;
}

void test_unreachable_upstream(Server& server) {
    std::cout << "Test: unreachable_upstream... " << std::flush;
    UpstreamOptions up;
    up.base_url = "http://127.0.0.1:" + std::to_string(DEAD_PORT);
    server.registerUpstream("dead", up);
    auto c = make_client();
    auto r = c.Post("/v1/chat/completions", R"({"model":"dead","messages":[]})", "application/json");
    assert(r && (r->status == 502 || r->status == 503));
    // 健康检查连续失败后标记为不健康：返回 503，且不出现在 /v1/models
    assert(wait_for([&] {
        auto info = find_upstream(server, "dead");
        return info && info->health == UpstreamHealth::Unhealthy;
    }));
    r = c.Post("/v1/chat/completions", R"({"model":"dead","messages":[]})", "application/json");
    assert(r && r->status == 503);
    r = c.Get("/v1/models");
    assert(r->body.find("\"dead\"") == std::string::npos);
    server.unregisterUpstream("dead");
    std::cout << "PASSED" << std::endl;
}

void test_admin_access_control() {
    std::cout << "Test: admin_access_control... " << std::flush;
    // 关闭时 404
    {
        Server s;
        ServerOptions o;
        o.host = "127.0.0.1";
        o.port = 18303;
        std::thread t([&] { s.run(o); });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        httplib::Client c("127.0.0.1", 18303);
        auto r = c.Get("/admin/upstreams");
        assert(r && r->status == 404);
        s.stop();
        t.join();
    }
    // 设置 key 后必须带 key；对外 API key 不影响管理接口
    {
        Server s;
        ServerOptions o;
        o.host = "127.0.0.1";
        o.port = 18304;
        o.api_key = "sk-client";
        o.admin_api = true;
        o.admin_key = "admin-secret";
        std::thread t([&] { s.run(o); });
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        httplib::Client c("127.0.0.1", 18304);
        auto r = c.Get("/admin/upstreams");
        assert(r && r->status == 401);
        r = c.Get("/admin/upstreams", {{"Authorization", "Bearer admin-secret"}});
        assert(r && r->status == 200);

        // 对外 API key 同样作用于上游模型
        UpstreamOptions up;
        up.base_url = "http://127.0.0.1:" + std::to_string(BACKEND_PORT);
        s.registerUpstream("qwen", up);
        r = c.Post("/v1/chat/completions", R"({"model":"qwen","messages":[]})", "application/json");
        assert(r && r->status == 401);
        r = c.Post("/v1/chat/completions", {{"Authorization", "Bearer sk-client"}},
                   R"({"model":"qwen","messages":[]})", "application/json");
        assert(r && r->status == 200);
        s.stop();
        t.join();
    }
    std::cout << "PASSED" << std::endl;
}

} // namespace

int main() {
    std::cout << "=== Upstream Tests ===" << std::endl;

    Backend backend;
    backend.start();

    Server server;
    UpstreamOptions up;
    up.base_url = "http://127.0.0.1:" + std::to_string(BACKEND_PORT) + "/";
    up.upstream_model = "";
    up.extra = {{"context", "extra"}};
    server.registerUpstream("qwen", up);
    assert(server.hasModel("qwen"));

    ServerOptions options;
    options.host = "127.0.0.1";
    options.port = GATEWAY_PORT;
    options.admin_api = true;
    options.upstream_health_interval = std::chrono::milliseconds(200);
    std::thread server_thread([&] { server.run(options); });
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    test_non_stream_passthrough(server);
    test_stream_passthrough(server);
    test_model_rename_and_upstream_key(server, backend);
    test_multipart_passthrough();
    test_upstream_error_passthrough();
    test_models_listing(server);
    test_admin_api(server);
    test_unknown_model_falls_through();
    test_client_disconnect_cancels_upstream(backend);
    test_unreachable_upstream(server);
    test_admin_access_control();

    server.stop();
    server_thread.join();
    backend.stop();

    std::cout << "\nAll tests PASSED!" << std::endl;
    return 0;
}
