# Nginx请求超时排查规范

当系统出现大量HTTP请求超时时，应首先检查Nginx错误日志。

重点关注upstream timeout、connection timeout以及upstream连接失败等错误。

如果CPU负载同时较高，应进一步检查后端应用进程是否存在资源竞争。

如果Nginx正常而后端响应时间过长，应继续检查数据库连接池、线程池以及后端服务运行状态。