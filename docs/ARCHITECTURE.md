# 系统架构与关键设计

## 1. 服务关系

```text
Vue 3
  │ JWT REST API / SSE
  ▼
FastAPI ────── MySQL（共享知识、个人数据、Agent检查点）
  │               ▲
  │ internal HTTP │
  ▼               │
Flask Collector ──┼── Redis/Celery ── Worker
                                  ├── 自动采集
Celery Beat ───────────────────────├── PDF/HTML解析
                                  ├── LLM结构化抽取
                                  ├── Milvus增量索引
                                  └── 日线行情同步
```

Vue不直接访问Flask、Redis、MySQL和Milvus。FastAPI是唯一公开业务入口。Flask使用内部令牌保护采集接口。

Milvus Standalone由Milvus、etcd和MinIO三个容器组成。MySQL保存业务真相与文本切片，Milvus仅保存用于检索的切片副本、标量过滤字段、稠密向量和BM25稀疏向量，因此向量集合可由MySQL安全重建。

## 2. LangGraph流程

```text
classify_question
  ├─ 系统问题 → answer_system_meta
  ├─ 非支持市场/范围外 → 说明能力边界
  ├─ 行业研究 → 跨公司混合检索
  └─ 公司研究 → resolve_entity
resolve_entity
  ├─ 公司不明确 → NEEDS_CLARIFICATION
  ├─ 不支持公司 → FAILED
  ▼
check_knowledge
  ├─ 缺失/最新数据过期 → start_collection → pause_for_collection → END
  ▼
retrieve_context
  ├─ 无证据 → handle_failure
  ▼
select_tools → analyze_evidence → validate_answer
  ├─ 部分来源缺失 → partial_answer
  └─ 资料完整 → generate_answer
```

采集完成后，Worker把同公司等待中的消息重新放入Agent队列。Graph重新检查知识库，随后继续检索与回答。业务恢复状态保存在`chat_messages`和`agent_checkpoints`中。

## 3. RAG与证据

- 文档按页解析，每片约700个中文字符，重叠100字符。
- BGE生成归一化的512维中文稠密向量，Milvus使用COSINE索引检索。
- Milvus通过中文分析器从切片正文生成BM25稀疏向量。
- 稠密语义检索与BM25关键词检索在Milvus中并行执行，通过RRF倒数排名融合。
- 公司、文档类型和时间范围在检索前通过Milvus标量表达式过滤，避免先召回错误公司的片段再过滤。
- 公司问题最多提供6个、行业问题最多提供8个证据片段。
- 公告和研报获得更高来源权重，近期资料获得适度时间权重，并限制同一文档重复占位。
- 每篇文档处理完后按`chunk_id`增量upsert；软删除文档时同步从Milvus删除，管理员也可从MySQL执行全量重建。
- 后台对比Milvus实体数与MySQL有效切片数并检查维度；连接异常时自动降级到本地FAISS/BM25，避免问答卡死。
- 引用保存文档ID、标题、来源类型、链接、日期、页码和原文。
- 实际财务数据和机构预测分表存储。
- LLM输出必须通过Pydantic验证；无法验证时退回规则抽取。

## 4. 数据和任务状态

文档状态：`DISCOVERED → DOWNLOADED → PARSED → EXTRACTED → INDEXED`，异常为`FAILED`。

问答主状态：`QUEUED → RESOLVING_ENTITY → PROCESSING → ANSWERING → COMPLETED/PARTIAL`。已有资料过期时，LangGraph先提交定向后台刷新，再继续RAG回答；刷新状态通过独立字段显示，不阻塞主状态。只有知识库完全为空时才走`COLLECTING → 自动续答`。另有`NEEDS_CLARIFICATION`、`CANCELLED`和`FAILED`，全部状态通过SSE实时推送。

采集来源支持`LIVE`、`SNAPSHOT`、`AUTO`三种模式。默认`AUTO`不以空结果覆盖历史数据。

## 5. 安全边界

- JWT有效期8小时；密码使用PBKDF2-SHA256加盐哈希。
- 管理端API检查`ADMIN`角色。
- Flask内部API检查`X-Internal-Token`。
- 不在代码或Git中保存模型密钥。
- 普通用户只能访问自己的会话、收藏和报告。
- Agent工具全部为研究型只读工具，不包含交易能力。
