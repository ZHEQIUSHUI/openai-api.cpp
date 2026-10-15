/**
 * OpenAI API Gateway - 统一网关
 *
 * 对外只开一个端口，按请求中的 model 字段把请求原样转发给已注册的上游服务。
 * 上游可以在启动参数里给出，也可以由管理平台通过 /admin/upstreams 动态注册。
 */

#include <openai_api/server.hpp>

#include <csignal>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace openai_api;

namespace {

std::unique_ptr<Server> g_server;

void signal_handler(int sig) {
    std::cout << "\nReceived signal " << sig << ", shutting down..." << std::endl;
    if (g_server) {
        g_server->stop();
    }
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program << " [options]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --host <addr>             Listen address (default: 0.0.0.0)" << std::endl;
    std::cout << "  --port <port>             Listen port (default: 8000)" << std::endl;
    std::cout << "  --api-key <key>           API key required from clients (default: disabled)" << std::endl;
    std::cout << "  --upstream <model>=<url>  Register an upstream at startup, repeatable" << std::endl;
    std::cout << "  --upstream-timeout <ms>   Read timeout for --upstream entries (default: 600000)" << std::endl;
    std::cout << "  --state <file>            Load upstreams from file at startup and save on every change" << std::endl;
    std::cout << "  --no-admin                Disable the /admin/upstreams API" << std::endl;
    std::cout << "  --admin-key <key>         Key required by the admin API (default: none)" << std::endl;
    std::cout << "  --admin-remote            Allow admin API from non-localhost addresses" << std::endl;
    std::cout << "  --health-interval <ms>    Upstream health check interval, 0 to disable (default: 5000)" << std::endl;
    std::cout << "  --threads <n>             HTTP worker threads, one per streaming client (default: 32)" << std::endl;
    std::cout << "  -h, --help                Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program << " --port 8000 --upstream qwen3-4b=http://127.0.0.1:18001" << std::endl;
    std::cout << "  curl -X POST http://127.0.0.1:8000/admin/upstreams \\" << std::endl;
    std::cout << "       -d '{\"model\":\"qwen3-4b\",\"base_url\":\"http://127.0.0.1:18001\"}'" << std::endl;
    std::cout << "  curl -X DELETE http://127.0.0.1:8000/admin/upstreams/qwen3-4b" << std::endl;
}

// 持久化用的完整配置（UpstreamOptions::to_json 不输出 api_key）
nlohmann::json upstream_to_state(const UpstreamInfo& info) {
    nlohmann::json j = info.options.to_json();
    j.erase("has_api_key");
    j["model"] = info.model;
    j["api_key"] = info.options.api_key;
    return j;
}

bool save_state(const std::string& path, const std::vector<UpstreamInfo>& upstreams) {
    nlohmann::json j = nlohmann::json::array();
    for (const auto& info : upstreams) {
        j.push_back(upstream_to_state(info));
    }
    const std::string tmp = path + ".tmp";
    {
        std::ofstream ofs(tmp, std::ios::binary | std::ios::trunc);
        if (!ofs) return false;
        ofs << j.dump(2) << std::endl;
        if (!ofs) return false;
    }
    std::remove(path.c_str());  // Windows 上 rename 不覆盖已有文件
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

bool load_state(const std::string& path, Server& server) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return true;  // 文件不存在：首次启动
    auto j = nlohmann::json::parse(ifs, nullptr, false);
    if (!j.is_array()) {
        std::cerr << "Error: invalid state file: " << path << std::endl;
        return false;
    }
    for (const auto& item : j) {
        if (!item.is_object() || !item.contains("model") || !item["model"].is_string()) continue;
        try {
            server.registerUpstream(item["model"].get<std::string>(), UpstreamOptions::from_json(item));
        } catch (const std::exception& e) {
            std::cerr << "Warning: skip upstream in state file: " << e.what() << std::endl;
        }
    }
    return true;
}

} // namespace

int main(int argc, char* argv[]) {
    ServerOptions options;
    options.port = 8000;
    options.admin_api = true;
    options.thread_pool_size = 32;
    std::vector<std::pair<std::string, std::string>> cli_upstreams;
    std::chrono::milliseconds upstream_timeout{600000};
    std::string state_file;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Error: " << name << " requires a value" << std::endl;
                std::exit(1);
            }
            return argv[++i];
        };
        try {
            if (arg == "-h" || arg == "--help") {
                print_usage(argv[0]);
                return 0;
            } else if (arg == "--host") {
                options.host = next("--host");
            } else if (arg == "--port") {
                options.port = std::stoi(next("--port"));
            } else if (arg == "--api-key") {
                options.api_key = next("--api-key");
            } else if (arg == "--upstream") {
                auto spec = next("--upstream");
                auto pos = spec.find('=');
                if (pos == std::string::npos || pos == 0) {
                    std::cerr << "Error: --upstream expects <model>=<url>, got: " << spec << std::endl;
                    return 1;
                }
                cli_upstreams.emplace_back(spec.substr(0, pos), spec.substr(pos + 1));
            } else if (arg == "--upstream-timeout") {
                upstream_timeout = std::chrono::milliseconds(std::stoll(next("--upstream-timeout")));
            } else if (arg == "--state") {
                state_file = next("--state");
            } else if (arg == "--no-admin") {
                options.admin_api = false;
            } else if (arg == "--admin-key") {
                options.admin_key = next("--admin-key");
            } else if (arg == "--admin-remote") {
                options.admin_allow_remote = true;
            } else if (arg == "--health-interval") {
                options.upstream_health_interval = std::chrono::milliseconds(std::stoll(next("--health-interval")));
            } else if (arg == "--threads") {
                options.thread_pool_size = std::stoi(next("--threads"));
            } else {
                std::cerr << "Error: Unknown argument: " << arg << std::endl;
                print_usage(argv[0]);
                return 1;
            }
        } catch (const std::exception&) {
            std::cerr << "Error: invalid value for " << arg << std::endl;
            return 1;
        }
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    g_server = std::make_unique<Server>();

    if (!state_file.empty() && !load_state(state_file, *g_server)) {
        return 1;
    }
    for (const auto& kv : cli_upstreams) {
        UpstreamOptions up;
        up.base_url = kv.second;
        up.read_timeout = upstream_timeout;
        try {
            g_server->registerUpstream(kv.first, up);
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
        }
    }
    if (!state_file.empty()) {
        auto save_mutex = std::make_shared<std::mutex>();
        Server* server = g_server.get();
        g_server->setUpstreamChangedCallback([server, save_mutex, state_file]() {
            std::lock_guard<std::mutex> lock(*save_mutex);
            if (!save_state(state_file, server->listUpstreams())) {
                std::cerr << "Warning: failed to save state file: " << state_file << std::endl;
            }
        });
        save_state(state_file, g_server->listUpstreams());
    }

    std::cout << "OpenAI API Gateway" << std::endl;
    std::cout << "==================" << std::endl;
    std::cout << "Listen: " << options.host << ":" << options.port << std::endl;
    std::cout << "API Key: " << (options.api_key.empty() ? "disabled" : "enabled") << std::endl;
    std::cout << "Admin API: "
              << (!options.admin_api ? "disabled"
                  : options.admin_allow_remote ? "enabled (remote allowed)"
                                               : "enabled (localhost only)")
              << (options.admin_api && !options.admin_key.empty() ? ", key required" : "") << std::endl;
    std::cout << "Upstreams:" << std::endl;
    for (const auto& info : g_server->listUpstreams()) {
        std::cout << "  - " << info.model << " -> " << info.options.base_url << std::endl;
    }
    std::cout << std::endl;

    g_server->run(options);
    return 0;
}
