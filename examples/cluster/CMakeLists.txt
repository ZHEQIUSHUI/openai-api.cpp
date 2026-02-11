# Cluster Mode Examples

# Master Server
add_executable(openai_api_example_cluster_master master.cpp)
target_link_libraries(openai_api_example_cluster_master PRIVATE openai_api::server)

# Worker Node
add_executable(openai_api_example_cluster_worker worker.cpp)
target_link_libraries(openai_api_example_cluster_worker PRIVATE openai_api::server)
