# CPED F/S/G 试标指南 v1.2

版本：`rcpem-public-pilot-v1.2`；日期：2026-10-22；任务：[Issue #3](https://github.com/Api-19/desktop-ai-companion-team/issues/3)。

状态：负责人已批准本次修订，后续批次采用 v1.2；校准批 1 的原始结果按 v1.1 保留，不重标、不跨版本混算。本文是试标规范，不是正式研究验收标准；新版定义的一致性待校准批 2 检验。已有 50 条材料草案保留，另增 8 条校准事件，按第 7 节分批冻结。标注使用负责人提供的离线标注页，页面导出 JSON；v1.2 工具另行更新，在工具支持并经主控通知前不开始新版正式标注。

### v1.2 相对 v1.1 的变化

- 保留 12 个动作，吸收校准批 1 的动作边界、预切合并处理和自由切分空白归属规则。
- 表达方式特征改为 mitigation、politeness、emotion、evaluation、rapport、humor、irony、intensification 共 8 项；evaluation 取五值，其余逐项三态。强化/夸张为候选项。
- seek_care 移为 REQUEST 子类型；self_disclosure 移为片段级内容属性，均保留采集。
- 收紧幽默证据，区分亲和拉近与一般关心、自我披露与实施动作；新增程序自动统计项，不自动换算人工标签。
- 新增校准批 2：从 remaining40 之外的合格 train 池选 8 条新事件，只做 S；与研究侧排除清单去重。原校准批 1 不重标，30 条验证批推迟到校准批 2 之后。
- 逐项报告正例、分歧、unclear 与状态一致率；evaluation 另报正负极性一致率，校准后逐项决定保留、修订或移出人工层。
- 完整保留适用于 v1.1 的 a2e-proj-v0；新增 a2e-proj-v1-draft 草案，待研究侧（RC-PEM）核对后定稿。
- 同步字段速查与独立编造的教学 JSON；本次不实现工具、不重标旧数据、不自动授权扩样。

## 1. 方法选择与依据

采用 **功能片段上的动作集合 + 片段顺序 + 独立表达方式特征**。分类对象是可见回复的交际功能，不是人格、隐藏动机或“正确回应”。同一语境可以有多种合理策略，历史实际回复只提供一种观察。

- [DIT++ / ISO 对话行为说明](https://dit.uvt.nl/)支持区分交际功能与其他维度，并允许话语具有多种功能。本指南借用结构原则，不声称实现 ISO 全集或标准兼容。
- [Welivita & Pu, COLING 2020](https://aclanthology.org/2020.coling-main.429/)区分追问、认可、建议、分享经历等共情回应意图，并在回复内部按句标注。其共情语料有领域限制，不直接整套移植。
- [Liu et al., ACL 2021 / ESConv](https://aclanthology.org/2021.acl-long.269/)区分提问、复述、反映感受、自我披露、宽慰、建议、信息等支持策略。只作为支持场景的定义参考，不把所有日常聊天都改成情绪支持。
- [Sharma et al., EMNLP 2020 / EPITOME](https://aclanthology.org/2020.emnlp-main.425/)分别标记共情机制并提供文本依据，启发本指南记录证据而不只给整轮单标签。
- [Bai et al., 2025 预印本](https://arxiv.org/abs/2505.15316)研究一轮中的连续多策略，支持保留动作序列；其情绪支持实验不等于本项目个人模仿效果证明。

以下 12 个具体动作和 8 个表达方式特征是本项目的试标设计，尚无证据表明这是唯一或最优分类。本次仅更新规范与格式例，不训练新模型、不改变线上策略或旧标注页面；内容属性与程序统计分别记录，不混入表达方式特征。

## 2. 材料、上下文与信息单元

### 2.1 一个事件是什么

一个事件是“可见前文 + 最后一个非目标说话人的话轮 → 目标说话人的完整下一话轮”。完整下一话轮包含同一人连续发送、直到下一人发言前的所有消息，保留消息边界与顺序。一次接话算一个事件，不把同轮拆成几条独立样本。

只取官方 train。使用来源 dialogue ID、原始行/utterance ID、角色 ID（带剧集命名空间）、源版本和抽样记录定位。正文不贴进 issue，记录来源划分。CPED 原 DA/情绪只供整理者筛查，保留在整理元数据里，不展示给独立标注 A/B，也不预填项目标签。

上下文从同一原始对话取目标话轮前最近最多 6 个完整话轮，必须含紧邻的对方话轮；不足 6 个取全部。一个话轮内的连续消息不截断。不跨对话补上下文，不看目标之后的发言。6 是本轮可复现的工作参数，不是文献结论。超长或缺失到无法理解的材料标记排除/待修复，记录原因；不能默默换成不同窗口。

记录 `context_truncated`、`missing_context`。有截断但当前问题仍可判断的样本可留；指代/角色/问题依赖缺失内容时用 `insufficient_context`。修复材料时先给两人同一新版本，保留原版本和修订理由。

### 2.2 F 信息单元与 S 功能片段分开

- **F 信息单元**：将上下文中可独立承接的事实、问题、感受、请求等划为有 ID 的单元。保留否定、时间、条件、施事和更正；不要按逗号机械切分。
- “明天要汇报，我有点紧张”可分成“明天要汇报”和“对此紧张”；“不是明天，是周五”整体保留为一个更正单元。
- F 信息单元只为预留的 10 条准备；其余 40 条不要求 F 单元。
- 单元由整理者在不看目标回复的条件下提出；如果已经看过，记录暴露状态，由未看目标的人复核。进入试标前统一单元版本。A/B 可以另提切分异议，不各用一套单元 ID。
- **S 功能片段**：对目标回复划分最小但足以判断交际功能的文本。可短于一句或跨连续消息；跨消息时保存多个 span，不拼改原文。不同动作能分开就分开；不可分的多功能片段允许多个动作，不能硬造先后。
- F 和 S 的单元集合、编号分开，禁止拿目标回复反推“上下文唯一正确切法”。

### 2.3 S 预切与独立切分诊断

整理者先按消息与标点生成候选边界，再按“功能”逐条核对，允许合并、拆分及跨消息片段，冻结为带 `s1, s2, …` ID 的预切片段。消息/标点分句不能直接当成功能切分。标注者只对每个预切片段标动作与特征，不得自行改片段；边界异议填写 `s.segment_objections`：`segment_id`、`problem`（`should_split / should_merge / wrong_boundary`）、一句 `proposal`，并选预设原因 `boundary`。需要合并时在 proposal 中指出另一片段 ID。

预切片段若包含可独立表达的连续功能，必须提交 should_split，在 proposal 中写明建议边界和各部分功能；不得将其当成“不可分多功能”。若当前片段无法用一个动作或合法的不可分动作集合表达，暂记 uncertain、reason_code=boundary，并在 reason 写明“可分功能被合并”；candidates 可留空，不用候选数组表示动作顺序。整理者统一重切后，双方按同版材料重新独立标注，原结果与重标结果分别保留和报告。例如“地点改到南门了。你这个安排很周到”应分为 INFORM→EVALUATE。

自由切分选择的是完整交际功能之间的插入位置，不是分词位置，也不是证据选区。“我赞同”不能切为“我｜赞同”；“我们再讨论一下”不能只因词语边界切开。消息两端不另点内部边界。功能边界附近若只有不承载功能的空白，统一将空白归右侧片段，边界点在空白之前，原文不改动。正式评分仍按消息 ID 与原始码点精确匹配，不自动移动已导出的边界。

校准批 1 的 **cped-001、cped-003、cped-007、cped-008** 已按 v1.1 做独立切分诊断，原始结果保留、不重做。诊断流程为：标注者先在未显示预切结果的原文上点选边界并提交，再进入预切片段标注。工具分别保存原始自由切分结果与预切标注，不以随后看到的预切结果覆盖诊断。码点偏移由工具生成，人工不填；校准批 2 不新增自由切分诊断。

报告分两项：①切分诊断（4 条，边界 precision/recall/F1）；②预切片段上的“条件一致性”（动作集合一致率等）。v1.0 的全量独立切分 F1 不再适用。

## 3. S 的 12 个动作

通常每个功能片段一个动作；无法分割且同时满足两个明确定义时可以多选，并写理由。若只是两种解释之间拿不准，用 `uncertain` 加候选，而不是多选。所有下列例子均为独立编造。

| 代码 | 名称与纳入条件 | 正例 | 边界 / 排除 |
| --- | --- | --- | --- |
| ACK | 接收/理解反馈：表示听见、收到、理解，不额外回答事实或作出行动承诺 | 对方叙述后说“原来如此” | 对“你明天来吗”回答“嗯”是 INFORM；对“明天帮我带书”说“好”是 COMMIT；不能仅凭词形判 ACK |
| INFORM | 回答/告知/解释：给出事实、经历、状态或原因，也含纠正事实 | “会议改到周五了”“我今天在实验室” | 个人经历仍属 INFORM，再加内容属性 self_disclosure；建议做什么不是 INFORM |
| ASK | 询问/澄清：确实希望对方提供或核实信息 | “你指的是哪份报告？” | 反问若表达评价或否定，标 EVALUATE；“能帮我拿一下吗”在索取行动，标 REQUEST |
| EVALUATE | 评价/立场：赞成、反对、判断好坏或表达偏好 | “这个安排不太合理”“你这次做得真好” | 拒绝具体请求用 REFUSE；认可对方感受的可理解性用 VALIDATE；单纯纠正日期用 INFORM |
| VALIDATE | 理解/认可体验：复述要点以核对理解，或明确承认对方感受和经历 | “所以你担心的是来不及？”“准备这么久还遇到变更，难怪会烦” | 真正询问未知原因用 ASK；“原来如此”仅反馈用 ACK；不从“嗯嗯”推断深层理解 |
| SUPPORT | 安慰/鼓励/宽慰：表达关切、陪伴或信心，以支持对方 | “辛苦了，先歇一会也没关系”“别灰心，可以慢慢来” | 具体下一步方案用 SUGGEST；要求对方来安慰自己不是 SUPPORT |
| SUGGEST | 建议/劝告：提出有助于对方的做法，包括叮嘱提醒 | “可以先问问负责人”“记得带伞” | 为完成说话者需求而要求对方行动用 REQUEST；不按祈使句/问句形式机械区分 |
| REQUEST | 请求/邀请：要求对方采取行动、给予资源/注意，或邀请共同活动；明确索取陪伴、关注、安慰或情感确认时标子类型 `seek_care` | “帮我看一下这页吧”“今晚一起吃饭吗？”；“陪我聊一会儿吧”为 REQUEST，subtype=seek_care | 真正索取事实用 ASK；明确为对方解决问题的做法用 SUGGEST；兼具两者且不可分可多标并说明；只说“我难受”不自动算 seek_care |
| COMMIT | 承诺/提供帮助：承担未来行动，或明确提出自己可提供的帮助 | “我明天帮你取”“要不我替你问一下？” | 单纯能力/日程陈述用 INFORM；仅建议对方自己行动用 SUGGEST；`subtype` 可选 promise 或 offer，也可留 null |
| REFUSE | 拒绝/设边界：拒绝请求、邀请或互动，要求停止越界行为 | “今晚不能去”“这件事我不想聊” | 不同意某个观点用 EVALUATE；“今天没去”只报告既往事实用 INFORM |
| SOCIAL | 社交/对话管理：问候、感谢、道歉、告别、明确开闭话题 | “谢谢你”“先聊到这吧” | 有实际内容的安慰不因礼貌而全部标 SOCIAL；subtype 可选，如 thank / apologize / greet / close / topic_shift，也可留 null |
| OTHER | 可识别但以上未覆盖的功能 | 如材料出现未覆盖功能，描述它做了什么 | 不等于看不懂。必须写功能描述，积累后决定是否增类 |

**不可判断是状态，不是第 13 个动作。** `status` 为 `judged / uncertain / insufficient_context`。后两种必须写原因；`uncertain` 可记候选动作，`insufficient_context` 不强行猜。`OTHER` 则属于 `judged`，必须有明确功能描述。

### 3.1 校准后的动作边界

**上下文不足与 OTHER**

判断动作只依赖当前可见内容中的交际功能。缺少人物关系或更早背景，但仍能判断是在评价、拒绝、告知时，不因此改为 insufficient_context。只有缺失信息会改变功能判断时，才记录相应不足。讽刺、反话等是表达方式；功能能归入现有类别时不使用 OTHER。例如，对方说“这次延期正好不用赶工”，回复“那倒省事了”，可以是 EVALUATE；不能仅因带反话色彩标 OTHER。

**可分功能被预切合并**

预切片段若包含可独立表达的连续功能，必须提交 should_split，在 proposal 中写明建议边界和各部分功能；不得将其当成“不可分多功能”。若当前片段无法用一个动作或合法的不可分动作集合表达，暂记 uncertain、reason_code=boundary，并在 reason 写明“可分功能被合并”；candidates 可留空，不用候选数组表示动作顺序。整理者统一重切后，双方按同版材料重新独立标注，原结果与重标结果分别保留和报告。例如“地点改到南门了。你这个安排很周到”应分为 INFORM→EVALUATE。

**INFORM / EVALUATE**

对事实、经历或自身状态问题作肯定、否定和解释，属于 INFORM；不能因为回复含否定词就改为 EVALUATE。对“你把附件删了吗”回答“没有，我只改了文件名”，是 INFORM；对“你还紧张吗”回答“不紧张了”，也是 INFORM。对方案、行为或观点作好坏、合理性等判断，才是 EVALUATE。

**EVALUATE / REFUSE**

REFUSE 必须能指出正在拒绝的请求、邀请、互动或越界行为。贬评对方能力、否定设想、反驳观点，不因发生在冲突中就自动标 REFUSE。事实或典故若主要作为反例来否定当前观点，整段可为 EVALUATE；若另有独立告知功能再切分。例如“他拿过奖，所以不会出错”→“冠军也会犯错”，在当前语境中主要是反驳保证关系，标 EVALUATE；“把车借我吧”→“今天不能借”，标 REFUSE。

**COMMIT / INFORM / SUGGEST**

COMMIT 要有明确承担未来行动或提供帮助的表达；承担未来不作为也可成立，例如“你决定报名，我保证不干涉”。仅表示未来可能改变想法、可能考虑某事，不自动构成承诺，例如“也许我之后会考虑换岗位”是 INFORM。向对方提出做法用 SUGGEST，例如“你可以先试一周”。条件句不自动是承诺；有条件但明确承担行动的“你资料备齐，我就替你递交”仍是 COMMIT。

**ASK / EVALUATE**

“是不是”不自动决定 ASK，“我觉得”也不自动决定 EVALUATE。先判断说话者是否确实要求对方提供或核实信息。例如“我不确定入口是不是东门，你帮我确认一下”是 ASK；“我觉得入口设在东门最方便”是 EVALUATE。若文字同时支持真实核实与立场表达，且没有足够证据排除任一解释，使用 uncertain 并记 ASK/EVALUATE 候选。

**subtype：**COMMIT 可选 `promise / offer`，SOCIAL 可选 `greet / thank / apologize / close / topic_shift / other`；这两类可留 null。REQUEST 明确索取说话者自身的陪伴、关注、安慰或情感确认时，填写 `seek_care`；普通事务请求或邀请留 null，不能从情绪陈述推断索取关照。同片段合法多动作需要子类型时，以动作代码为键分别存，如 `{"REQUEST":"seek_care"}`。seek_care 不再放在 modifiers，也不是第 13 个动作。

## 4. 表达方式特征

### 4.1 八个可并存的表达方式特征

每个片段的 8 项都必须逐项显式选择，不从动作自动推导，不打强度分，不推断真实心情、人格或人物关系。除 evaluation 外，取值为 `present / absent / unclear`。`present` 点选证据，由工具记录原文定位；`unclear` 写短原因；`absent` 不写证据。界面不得默认 absent，未选视为未完成；片段功能不可判断时仍逐项检查，无法判断的项选 unclear 并说明原因。

`evaluation` 单独取 `positive / negative / mixed / absent / unclear`，不用 present：positive、negative、mixed 都属于“present 类取值”，必须点选证据；mixed 须分别定位褒贬证据。unclear 写短原因，absent 不写证据。absent 只表示没有达到该项证据标准，不代表相反风格，也不给整句打礼貌或情绪分。以下正反例均为独立编造；反例只说明该项 absent，不限制其他项。

| 字段 / 名称 | 定义与最低证据 | 正例 | 反例 | 边界与理论来源 |
| --- | --- | --- | --- | --- |
| `mitigation` 缓和/认识保留 | 用“可能、大概、我觉得、要不、好像”等降低断言确定性或要求力度，能指出被缓和的断言或要求 | “要不先检查一页？”中“要不”给建议留选择空间 | “我觉得头疼。”中“我觉得”只报告身体感受，不起保留作用 | 不凭词形判定；真实时间条件、问句形式不自动算缓和。只报告客观概率而未体现保留作用也不自动算。模糊限制语与 mitigation [M1–M2]、礼貌理论 [P1]；本项目合并两类可观察用法，不声称理论上等同。 |
| `politeness` 礼貌缓冲 | 可明确识别的敬让、致歉、感谢或减轻强迫感的表达，如“麻烦你”“谢谢”“不好意思” | “不好意思，麻烦您再发一次。” | “把附件发来。” | 没说“请”且没有其他证据时为 absent，不因此推断无礼；不评整句礼貌程度。明显反话不能只凭“谢谢”等字符串标 present。礼貌理论与计算礼貌 [P1–P3]。 |
| `emotion` 情绪外显 | 措辞明显传达说话者自身的喜、怒、悲、惧、担忧等，只标外显表达 | “我真的气死了！” | “预约被取消了。” | 坏事或好事发生、他人的情绪、感叹号本身不够；不猜真实内心。情绪标注 [E1–E2]；不沿用原数据情绪预填答案。 |
| `evaluation` 评价立场 | 对人、行为或事物明确褒贬；positive=褒，negative=贬，mixed=同片段兼有两者 | “这个排版清楚又整齐。”为 positive | “这份材料有六页。”为 absent | 纯事实、否定词、自身情绪不自动构成评价；“版式清楚但配色刺眼”为 mixed。不与动作 EVALUATE 相互自动补标。评价理论 Attitude [A1–A2]。 |
| `rapport` 亲和拉近 | 用昵称、“咱们”、友好称谓或逗趣称呼主动拉近距离，适用于任何关系 | “搭档，咱们把这关过了吧。” | “忙完早点休息。” | 关心健康、提醒休息、礼貌照顾不自动成立；只有“你/我”或基本礼貌形式不算。昵称、称谓须在语境中起拉近作用，不能用已知关系或“很体贴”代替文本证据。取代旧 affection，范围扩展到一般亲和表达，不推断关系阶段。积极礼貌与汉语礼貌 [P1–P2]。 |
| `humor` 戏谑幽默 | 有可辨识的玩笑、双关、荒诞或预期反转，并能指出玩笑构造 | 明确玩笑语境中：“我的打印机今天兼职碎纸机，专挑关键页罢工。” | “哈哈，材料收到了。” | “哈哈”、夸张、成语、典故、否定、冒犯均不能单独证明幽默；不要求善意、亲密或标注者觉得好笑。认真与玩笑都有依据且无法分辨时 unclear。幽默研究 [H1]，详见下方校准边界。 |
| `irony` 反讽/反话 | 字面立场与可见语境支持的实际立场明显相反 | 约好三点，对方四点才到：“你可真准时。” | 同一迟到语境中：“你晚了一小时。” | present 须同时点选字面表达和反向语境依据，并用短 reason 说明相反立场；无法确认时 unclear。反问、否定、批评不自动算反话；与 humor 可同现。反讽标注与语境研究 [I1–I2]。 |
| `intensification` 强化/夸张【候选】 | 明显强化、极端化或不按字面理解的夸张，能定位强化的措辞与对象 | “我等到天荒地老了。” | “我等了比较久。” | 普通“很、比较”不算；夸张不自动等于 humor，也不把标点数量换算为强度。**候选项：校准批一致性不足则移出人工层，改为程序统计。**评价理论 Graduation [A1]。 |

**humor 的校准证据边界：**

humor 的证据必须支持可辨识的玩笑、戏谑或搞怪表达，不能只指出否定词、成语、典故、语气词或不确定词。严肃拒绝、贬斥、反驳和警告不会因措辞夸张而自动 present；明确的戏谑也不要求善意或亲密。语境支持认真与玩笑两种读法、仅凭文字无法区分时选 unclear。例如在明确玩笑语境中说“你这速度，蜗牛都要给你让路了”可为 present；严肃说“这条件根本没希望”不因夸张而成为 present。

因此不把“调侃”和“撒娇”当成与问答、建议互斥的动作；表达方式、REQUEST 子类型 seek_care 与内容属性各自记录。旧 A–E 原始结果保留旧版本，派生投影见第 11 节。

**理论来源（离线审阅中书目身份标为“确定”的文献）：**下列仅为定义与证据写法的依据，不表示八项已在中文聊天中验证，也不声称整套实现原理论；本次未联网核验文献原文或一致性数字。

- [P1] Penelope Brown、Stephen C. Levinson，1987，*Politeness: Some Universals in Language Usage*，Cambridge University Press。
- [P2] Yueguo Gu，1990，〈Politeness phenomena in modern Chinese〉，*Journal of Pragmatics*。
- [P3] Cristian Danescu-Niculescu-Mizil、Moritz Sudhof、Dan Jurafsky、Jure Leskovec、Christopher Potts，2013，〈A Computational Approach to Politeness with Application to Social Factors〉，ACL。
- [M1] George Lakoff，1973，〈Hedges: A Study in Meaning Criteria and the Logic of Fuzzy Concepts〉，*Journal of Philosophical Logic*。
- [M2] Bruce Fraser，1980，〈Conversational mitigation〉，*Journal of Pragmatics*。
- [E1] Yanran Li、Hui Su、Xiaoyu Shen、Wenjie Li、Ziqiang Cao、Shuzi Niu，2017，〈DailyDialog: A Manually Labelled Multi-turn Dialogue Dataset〉，IJCNLP。
- [E2] Dorottya Demszky、Dana Movshovitz-Attias、Jeongwoo Ko、Alan Cowen、Gaurav Nemade、Sujith Ravi，2020，〈GoEmotions: A Dataset of Fine-Grained Emotions〉，ACL。
- [A1] J. R. Martin、P. R. R. White，2005，*The Language of Evaluation: Appraisal in English*，Palgrave Macmillan；Attitude 与 Graduation 分别启发评价立场与程度强化的区分。
- [A2] Jonathon Read、David Hope、John Carroll，2007，〈Annotating Expressions of Appraisal〉，*Proceedings of the Linguistic Annotation Workshop*。
- [H1] Salvatore Attardo、Victor Raskin，1991，〈Script theory revis(it)ed: joke similarity and joke representation model〉，*Humor: International Journal of Humor Research*。
- [I1] Cynthia Van Hee、Els Lefever、Véronique Hoste，2018，〈SemEval-2018 Task 3: Irony Detection in English Tweets〉，*Proceedings of the 12th International Workshop on Semantic Evaluation*。
- [I2] Byron C. Wallace、Do Kook Choe、Laura Kertz、Eugene Charniak，2014，〈Humans Require Context to Infer Ironic Intent (so Computers Probably do, too)〉，ACL，短论文。

**中文标注证据（主人用联网检索核对，2026-10-22）：**下列数值是相邻任务的人工一致性，定义、语料和粒度与本指南不同，只说明这些维度在中文上可以操作化，不代表本指南的标签已达到同样一致性。

| 维度 | 研究 | 语料与标注 | 原文一致性 |
| --- | --- | --- | --- |
| 礼貌（整体程度） | Li 等 2020，CSCW（Studying Politeness across Cultures Using English Twitter and Mandarin Weibo） | 5,300 条微博，每条 2 名标注者，七级评分 | Krippendorff α = 0.661 |
| 礼貌（整体程度） | Havaldar 等 2023，EMNLP（Comparing Styles across Languages） | 5,700 条中文 Wikipedia Talk，每条 3 人，五级 | Fleiss κ = 0.206 |
| 回应性情绪 | Hu 等 2024，Findings of ACL（RESEMO） | 微博主帖与评论，16 类 | Cohen κ = 0.5119 |
| 对话情绪 | Sun 等 2026，Findings of ACL（EmotionTalk） | 普通话对话，7 类，仅文本 | Fleiss κ = 0.66 |
| 反讽 | Xiang 等 2020，LREC（Ciron） | 约 8,700 条微博，5 人，五级 | Fleiss κ = 0.470 |
| 幽默（笑话） | Zhang 等 2019，EMNLP-IJCNLP（Telling the Whole Story） | 9,123 个中文笑话，600 句 3 人复标 | 类别 κ = 0.71，程度 κ = 0.65 |
| 缓和语（英文） | Wang 等 2026，PLOS ONE | 中国学者英语学术口语，2 名编码者 | Cohen κ = 0.85 |

可用于 §4.3 自动统计的中文资源：简体中文 LIWC（SCLIWC2024，崔雪婷等 2024）、中文礼貌词库 Chinese PoliteLex（Li 等 2020）、大连理工大学情感词汇本体（徐琳宏等 2008）。

### 4.2 内容属性

`self_disclosure` 是片段级内容属性：提供说话者自己的经历、状态、感受或偏好，包括明确的接受界限。取值 `present / absent / unclear`，必须显式选择；present 点选证据，unclear 写短原因，absent 不写证据。它不计入上面的 8 个表达方式特征，不从 INFORM 等动作自动推导。

“我以前也遇到过”可标 INFORM + content_attributes.self_disclosure=present；“我觉得你可以问问”里的“我觉得”不是自动自我披露。

第一人称不是自我披露的充分条件。“我建议先排队”只是在实施建议，不自动 present；“我上周用过这个工具”报告自身经历，属于 present。明确表达自己的接受界限或偏好也属于 present，例如“只要不用熬夜，其他安排我都能接受”。单纯“我替你预约”是在承诺行动，不因出现“我”就自动添加该特征。

### 4.3 程序自动统计项（不人工标注）

以下只约定后续工具的统计范围，本次不实现。**统计用于作者层聚合，不自动换算成人工标签。**词表、分词、归一化与窗口规则须固定版本，保留同形异用和极短回复的限制；缺少可靠时间信息时不推断语速或延迟。

| 统计组 | 口径与边界 |
| --- | --- |
| 句末语气词与叹词 | 统计候选词形、位置与组合，区分同形异用；不换算为亲密、缓和或情绪。 |
| 人称与功能词 | 使用 SCLIWC2024 词典类别，固定实际词典及分词版本；不直接套用英语类别或心理结论。 |
| 回复长度与消息条数 | 按原始完整话轮、原消息边界统计长度与消息条数，跨事件汇总分布；不把长短当人格。 |
| 标点/表情/颜文字 | 统计类型、频次与连续形式；不可见贴图只计占位符，不猜含义。 |
| 模糊限制语与强化词词频 | 固定词表统计候选次数与频率；出现“可能”“很”不自动生成人工 mitigation 或 intensification。 |
| 字词重复 | 统计连续重复、拖长等形式，保留原文，不直接解释为强调或情绪。 |
| 口头禅 n-gram | 跨多条回复统计重复字词串及句首句尾位置；控制模板和话题词影响。 |
| 互动节奏 | 按固定窗口统计短回复比例、连续追问；短回复阈值和追问识别规则由统计工具另行固定，不要求人工新增标签。 |

跨作者比较需控制动作、场景、对象和来源差异；单条形式出现不等于稳定个人习惯，自动统计也不替代逐项证据判断。

## 5. 边界练习（公开教学例，不作未见测验）

| 可见语境 → 回复 | 示范标法与理由 |
| --- | --- |
| “我又迟到了。”→“你这是和闹钟有仇吧。” | EVALUATE + humor；不是索取真实信息，不因问句形式标 ASK |
| “我准备放弃。”→“你是担心时间不够，还是不喜欢这个方向？” | ASK；提出备选原因请求核实，不直接宣称理解成立 |
| “我担心时间不够。”→“所以主要卡在时间，对吗？” | VALIDATE；复述已有要点并核对理解，不是问新事实 |
| “今天好累。”→“我也是。” | INFORM + 内容属性 self_disclosure；可表达共同体验，但没有充分证据就不自动补 SUPPORT |
| “今天好累。”→“辛苦了。要不要先睡会儿？” | 两段 SUPPORT → SUGGEST；后句主要建议休息，不是调查是否困倦 |
| “今晚一起吃饭吗？”→“好呀。” | COMMIT，promise；实际接受并承担共同行动，不是 ACK |
| “这个方案最省钱。”→“我不同意，这个还没算运费。” | EVALUATE → INFORM；反对观点并给出事实依据 |
| “帮我写完这份报告。”→“这部分需要你自己完成。” | REFUSE；此处在拒绝请求，不只按句式标建议 |
| “能陪我一会儿吗？” | 有相应语境时 REQUEST，subtype=seek_care；不因为撒娇口气而覆盖请求功能 |
| “我以前也卡在这里，确实很折磨。可以先把问题拆小。” | INFORM(content_attributes.self_disclosure=present) → VALIDATE → SUGGEST；保留整轮顺序 |
| “你先忙。”→“那我可要收陪聊费啦。” | 上下文足以认定玩笑时可为 REQUEST + humor；无法判断是真要求还是玩笑时 uncertain，记录候选与疑点 |
| 只有“[某表情]”，无可见图像或稳定释义 | insufficient_context；不能凭猜测生成表情含义 |

旧的“开头嗯嗯一律跳过”取消：ACK 如实际出现则保留。若只做首动作基线，必须另行派生并说明规则，不覆盖完整标注。

### G 对照例

**均为编造，教学用，不计一致性。**`x` 为当前语境，`r` 为历史候选回复；判定口径见第 6 节。

| # | 当前语境 x | 候选回复 r | 示范判定与理由 |
| --- | --- | --- | --- |
| 1 | 想把抽屉重新分区，一整排看着无从下手 | 先挑最简单的一小段试试，不用一下全弄好（原话题是织围巾） | applicable：跨话题，“刚开始、无从下手”的前提成立。 |
| 2 | 抽屉都理好了，给你看成果 | 先挑最简单的一小段试试，不用一下全弄好 | not_applicable：状态已经完成。 |
| 3 | 这次没掉链子的是他，我那段还是忘词了 | 稳稳讲完啦，给你鼓个掌 | not_applicable：人不对，状态也相反。 |
| 4 | 卡扣我不会扣，先别教我，陪我吐槽两句 | 拍近一点给我，我帮你看看卡在哪 | not_applicable：违背她的明确要求。 |
| 5 | 周末搬绿植的事，咱俩之前怎么说的来着？ | 记着呢，周末我过去帮你搬 | uncertain：旧承诺可能有也可能没有，两种补全都说得通。 |
| 6 | 快递到了，东西也齐，就跟你报一声 | 就问客服现在到哪一步了 | not_applicable：状态和需求都对不上。 |
| 7 | 刚拼完拼图，想拍给你看 | 快发来，让我看看你的成果（原话题是练舞） | applicable。 |

## 6. F 和 G 的执行规则

### F：可接受焦点与实际焦点分开

1. **先锁定第一步**：A/B 只看同版上下文与 F 单元，选可接受单元集合。`preferred` 为某个已选单元、`no_unique` 或 `insufficient_context`。允许选多个；整体承接时选择被整体承接的单元集合，并记 `mode=holistic`。第一步结果封存后才能看回复。
2. **第二步**：揭示完整目标回复，记录实际承接的单元及首次承接顺序。能映射时填单元 ID；同时承接可并列，不硬排。没有承接任何已列单元时记录 `no_match`，上下文不足则记录 `insufficient_context`，两者不同。
3. 不回改第一步以迎合真实回复。材料整理者若已看过回复，不能把自己的 F 第一步结果算作盲标；每名标注者的 `reply_seen_before_step1` 由批次/事件暴露记录和页面流程生成，已暴露记录只作为开放复核。
4. 单元有缺漏时提异议，先保存当次原答案。调整单元后，对受影响事件按新版本重标，分别报告原始一致性与修订结果。

### G：只在有合规候选时做

G 以“一个事件 × 一个候选历史片段”为一对。候选限目标说话人在同一原始对话内、目标话轮之前的已有发言及其必要前文；先不跨剧集/角色拼历史。不能编造真实历史凑数。

候选 = 目标说话人过去的一段（对方当时的话 c → 他的回复 r），用作回应方式和措辞的参考，不是事实来源。判断：把 r 的回应方式用到当前语境 x，r 默认的前提——事情所处的状态、对方的需求或明确要求、所指的人、旧事实或旧承诺——在 x 下是否都成立。话题不同本身不算冲突。

- applicable：前提都成立。
- not_applicable：至少一个前提和 x 冲突；r 里带有 x 中没有、也无法成立的旧事实或旧承诺，同样判 not_applicable。
- uncertain：x 缺少关键信息，存在两种同样合理、但会让结论相反的补全时才用；仅仅“没提到”不算。
- 公开数据没有许可信息，只判内容；称呼和调侃的许可由研究侧代码规则负责，标注时不考虑。

记录简短理由和证据。每个候选保留 c、r 与当前上下文 x 三者的来源消息 ID：`c_message_ids`、`r_message_ids`、`x_message_ids`。人工 G 只看上下文与候选，不看当前目标回复；应在 F 第一步锁定之后、揭示回复之前完成。已看过目标回复者记录暴露，结果另报。每个事件实际有几个候选就报几个，不将事件数当配对数。事件本身没有候选时用事件级 `g_status=not_applicable` 和空列表，不算 uncertain；有候选时 `g_status=annotated`，各候选按上述三类判断。

G 人工标签将用作研究侧门控模型的人工验证集，用于 G 的事件必须不在研究侧弱标签训练源对话清单中（清单由研究侧持有，只含 TV_ID+Dialogue_ID，负责人核对）。本轮预留的 10 条已按此核对；本文不列具体对话 ID。

## 7. 首轮流程与交付

### 7.1 分批与进入条件

整理者按本指南核对现有 50 条的来源/权利条件、同版窗口、完整目标话轮、功能预切、无未来信息与无重复，并填写第 10 节覆盖矩阵；未核实项列出，不默认通过。不给 A/B 预填原 DA/情绪、模型建议或教学答案。

预留 F/G 批的 10 条为：**cped-013、cped-022、cped-027、cped-032、cped-034、cped-040、cped-042、cped-043、cped-044、cped-047**。在 F/G 完成前，两人都不得看这 10 条的目标回复；此处指完成揭示前的 F1 与 G，随后按流程揭示。材料包中这 10 条的 stage2 单独封存，不随验证批发放，也不得从其他材料泄露。

| 阶段 | 做什么 | 进入下一阶段的条件 |
| --- | --- | --- |
| 材料冻结 | 核验基础 50 条，冻结预切和版本；预留 10 条及相关候选单独封存。 | 同版任务包可还原；许可适用范围明确；已知材料问题有处理结果；主控明确通知开标。 |
| 短练习 | 共同看 4—6 个合成边界例，练习页面操作与 uncertain/多功能区别。 | 两人能正确操作并说明核心规则；练习不计一致性。 |
| 校准批 1（已完成，v1.1） | 原首批 10 条只做 S，指定 4 条的独立切分诊断一并保留。原始结果按 v1.1 保存，不按 v1.2 重标。 | 原始结果与讨论/裁决分离；本次修订不能回写提高旧一致率。 |
| 校准批 2（v1.2） | 整理者从 remaining40 之外的合格 train 池选 8 条新事件，与原 50 条无重复，并由研究侧核对排除清单去重；冻结新任务包，只做 S（动作、8 项表达方式、内容属性及适用 subtype），不安排 F/G。 | 独立结果分别保存，完成分歧归因；按第 8 节逐项给出保留 / 修订 / 移出人工层的结论，主控确认后再进入验证批。 |
| 验证批（推迟至校准批 2 后） | remaining40 中除预留 10 条外的 30 条，只做 S；按校准批 2 后正式冻结的版本执行。 | 原始分母、逐类支持数、特征分布和主要分歧可报告；不靠单一总一致率放行。 |
| 预留 F/G 批 | 上列 10 条：F1 锁定 → G → 揭示目标回复 → F2 → S；先取约 10 个合规候选对，没有候选则明确 G 未实施。 | 分别报告 F/G 是否可执行及其主要缺口，不要求三层同时“达标”。 |
| 扩样决策 | 主控验收后决定是否取 200 个新事件；定义仍不稳定时只补一个小批验证。 | 工具可用、规则主要歧义得到处理、覆盖缺口可解释、实际工时可接受；不以讨论后一致率替代独立结果。 |

校准批 1 的 v1.1 原始结果永久保留，本次不重标；未采集的新字段记为“未采集”，不得补为 absent。后续从 v1.2 开始；若校准批 2 后还需改定义，再冻结新小版本。需要桥接时另行指定子集，不覆盖旧结果、不跨版本混算，更改规则后看过的材料不再声称未见。新增 8 条校准事件单列批次和分母，不计入原基础 50 条或第 10 节最多 10 条的补缺额度。

### 7.2 暴露声明与保存

每名标注者每批提交一条开始前的 `batch_declaration`：`reply_seen`（是否看过本批任何目标回复）、`review_comments_read`（是否读过 Issue #3 中对本批材料的核验评论）。逐事件例外用 `event_exposure_exceptions`，每项含 `event_id`、`reply_seen`、`review_comments_read`、`note`；例外覆盖该事件的批次声明，用于保留事件级差异。后续发现暴露也须补记。B 已读过首批 10 条的核验评论，F/G 不在首批做，因此只记录不剔除。

每人每批独立导出一个 JSON，禁止使用同一路径覆盖他人结果；不再要求手写 JSON。[annotation_record.example.json](annotation_record.example.json) 是独立编造的教学记录，含示范答案，`phase=training`，不能混入盲标材料包。旧 Admin 单标签策略页和 `m1_labeler.py` 的 A–E 假设不能完整承载本协议，不能把 12 类硬塞旧页面或脚本。现有离线标注页、导出校验器及配套合成任务包仍为 v1.1，不能直接消费本 v1.2 格式例；新版工具与任务包另派更新，本契约不实现工具。

导出顶层固定 `protocol_version`、`material_version`、`annotator_id`、`phase`、`batch_id`、`batch_declaration`、`event_exposure_exceptions`、`records`。匿名标注者 ID 即可；`phase` 为 `training / pilot / adjudication`。来源定位、文本版本哈希在获准任务包 manifest 中自动维护。S 片段的 `spans` 使用 `message_id / start / end / text`，偏移按原消息 Unicode 码点、0 起始、左闭右开，不按字节或 UTF-16；`text` 必须等于原切片。数组保留片段出现顺序，跨消息片段用多个 span，`actions` 数组内部无先后。

正常 `judged` 片段不写理由；只有 `uncertain`、`insufficient_context`、`OTHER`、同片段多动作、切分异议时写片段级理由；特征/内容属性的 unclear 理由与 irony 的反向依据说明另按第 4 节填写。异常片段先选 `reason_code`，必要时再补一句 `reason`。预设原因：`context_missing / ambiguous_intent / joke_or_serious / multi_function / boundary / other`。OTHER 的补充说明必须描述功能；切分异议以一句 proposal 说明改法。未判断片段 `actions=[]`，`candidates` 另存候选。COMMIT/SOCIAL 的 subtype 可留 null，首轮不统计这两类 subtype；REQUEST 满足 seek_care 定义时必须填写，在校准批 2 单列其正例和分歧，不能因未标可选 subtype 而漏采关照请求。

原始 A/B、AI 建议（如以后获准使用）、裁决结果三个版本分开存。裁决不能改写原始一致性；保留争议、改动、理由、裁决者与版本。任务包哈希、指南版本和来源版本必须可追溯。正文材料与结果按获准方式单独交付；公共仓库只交规范、合成示例和不含正文的来源/切分清单，不上传真实 CPED 原文或完整标注包。

### 7.3 首轮工时预算

v1.1 原 50 条的每人约 3.5—5 小时预算仅作历史参考，不能直接作为 v1.2 的工时承诺。v1.2 增至 8 项表达方式及 1 项内容属性，另有 8 条校准批 2；复杂证据定位和反话核对也需计时。

校准批 2 分别记录每人规则阅读、逐事件/片段标注、证据定位、边界异议和分歧讨论的实际用时，再据此更新 30 条 S 验证及预留 F/G 的预算。原校准批 1 不计重标工时；材料核验、预切与工具准备属主控侧工作，另计成本。本次没有新版实测，不填入未经验证的增时数字。

## 8. 怎样判断试标是否可用

首轮是规范检验，不设“达到某分数即可上线”的门槛。至少交付：

- **有效分母**：基础 50 条、新增 8 条校准批 2 与补缺分开；按批次报已完成事件、片段、排除/缺失/已暴露数量，F/S/G 分别有多少有效判断；未补齐时明确实际数。
- **S 条件一致性**：在同版预切片段上报动作集合完全一致率、逐类分歧与支持数，并报共同可判断片段数及全部预切片段数；不可判断数量与 OTHER 原因单列，不能静默丢掉难例。
- **8 项表达方式特征**：每项分别报 A/B 的 present 正例数、正例相关分歧（至少一方 present 但状态不同）、unclear 数与状态一致率，注明有效分母，未采集不作 absent。evaluation 的 positive / negative / mixed 计为 present 类：分列三类支持数、其并集正例数及至少一方为 present 类而五值不同的分歧；状态一致率按五值精确相同计算。
- **evaluation 极性一致率**：positive 与 negative 各自另报二元一致率；mixed 同时具有正、负极性，absent 两者均无。任一方 unclear 的配对不进极性一致率分母，单列数量；同时报至少一方具有该极性的子集一致率与分母，避免大量双 absent 掩盖分歧。
- **内容属性与关照请求**：self_disclosure 独立报告 present、unclear、状态一致率和正例相关分歧；REQUEST subtype=seek_care 单列双方正例数、至少一方命中时的分歧及有效分母，不计入 8 项表达方式特征。
- **逐项处置**：校准批 2 结束时，8 项逐项写“保留 / 修订 / 移出人工层”、支持数与理由；没有正例不能宣称定义已稳定。intensification 一致性不足则移出人工层、改为程序统计；如修订取值或移出，先冻结新版本再用于验证批，不把未采集字段补为 absent。
- **切分诊断**：只对 cped-001、cped-003、cped-007、cped-008 的两份原始边界报告 precision/recall/F1；按 message ID 与 Unicode 码点位置精确匹配，忽略消息两端边界，以 A 为参照计算并注明方向。另报双方边界数；双方都无内部边界时单列，不记虚假满分。
- **边界异议**：按批次报切分异议数及 should_split / should_merge / wrong_boundary 分布。预切子集与自由切分子集分开报告，同一诊断事件随后做的预切标注仍属条件一致性，不能混作独立切分成功。
- **F 子集**：第一步报可接受集合逐事件 Jaccard、首选一致率，no_unique 与不足单独报；两空集合不当作满分。第二步报承接集合重叠与顺序差异，单列 no_match。
- **G 子集**：报实际事件×候选数、三类分布与一致率。只有“事件本身没有候选”（`g_status=not_applicable`）不进入配对分母；候选存在但被判 not_applicable 的必须计入分母。候选未准备时明确 G 未实施，暴露结果另报。
- **分歧归因**：至少按上下文不足、切分不同、定义重叠、特征证据不足、漏类归因，原始独立结果与讨论/裁决结果分别报告。

κ 与动作序列归一化编辑距离可后补；有足够类别变化才计算 κ，单一类别记不可计算。个人策略研究后续需比较相似语境下的动作选择、组合、顺序和互动特征。本批跨角色试标只能验证标注可行性，不能证明已学会某个人，不能把“更合适/更会安慰”当成“更像本人”；不收集私人聊天、个人习惯卡、关系推断或主人的私有评分。

## 9. 记录字段速查

以下是页面导出约定，格式例与本表同步；元数据、偏移与来源 ID 由工具带出，人工只做判断。

| 字段 | 必填内容 / 约束 |
| --- | --- |
| protocol_version / material_version / annotator_id / phase / batch_id | 协议版本、冻结材料版本、匿名标注者、training / pilot / adjudication、批次 ID |
| batch_declaration | reply_seen、review_comments_read 两个 bool，每人每批一条 |
| event_exposure_exceptions[] | event_id、reply_seen、review_comments_read、note；无例外为 [] |
| record.event_id / material_ref / sample_role | 同一冻结任务包事件 ID、manifest 引用；sample_role 为 base 或 supplement；训练示例可内嵌 training_material，正式输出不复制正文 |
| f_step1 | locked、reply_seen_before_step1、status、acceptable、preferred、mode、reason；status 为 judged 或 insufficient_context；不足时 acceptable=[]、preferred=insufficient_context；无唯一首选不等于不足；S-only 批为 null |
| f_step2 | status、mode、responded_groups、reason；status 为 judged / no_match / insufficient_context；responded_groups 按首次承接顺序存 ID 数组，同组表示同时承接，重复只记第一次；无匹配/不足时为空；S-only 批为 null |
| F mode | units 或 holistic；holistic 仍列出所承接的 ID，整段笼统回应无法定位时说明原因，不强迫全部命中 |
| g_status / g_reason | annotated 或 not_applicable；前者至少一个 g_pair，后者无候选、列表为空并说明原因；S-only 批为 null，表示未安排 G，不等同无候选 |
| g_reply_seen_before_judgment | G 判断前是否已看过目标回复，由暴露记录/流程生成；未做 G 时为 false |
| g_pairs[] | candidate_id、c_message_ids、r_message_ids、x_message_ids、label、reason、evidence；label 只取 applicable / not_applicable / uncertain；三组 ID 分别定位历史对方话 c、历史回复 r、当前上下文 x；evidence 为其中的消息 ID，不依赖当前真实回复 |
| s.segments[] | 冻结的 id（s1, s2, …）、spans、status、actions、candidates、subtype、modifiers、content_attributes、reason_code、reason；按原文次序，不遗漏 ACK/SOCIAL；status 为 judged / uncertain / insufficient_context，后两者 actions=[] |
| s.segment_objections[] | segment_id、problem（should_split / should_merge / wrong_boundary）、一句 proposal、reason_code=boundary；不得自行修改冻结片段 |
| subtype | COMMIT 可选 promise / offer、SOCIAL 可选 greet / thank / apologize / close / topic_shift / other，两者可留 null；REQUEST 明确索取自身陪伴、关注、安慰或情感确认时必须填 seek_care，其他请求留 null；多动作需细分时以动作代码为键分别存 |
| modifiers | 恰好 mitigation / politeness / emotion / evaluation / rapport / humor / irony / intensification 八键，各有 value；除 evaluation 外取 present / absent / unclear；evaluation 取 positive / negative / mixed / absent / unclear。present 及评价的三种 present 类值带 evidence（message_id / start / end / text 定位数组），unclear 带 reason，absent 只存 value；irony=present 的 evidence 同时定位字面片段与可见反向语境，并以 reason 说明；必须逐项显式选择 |
| content_attributes.self_disclosure | 片段级内容属性，有 value=present / absent / unclear；present 带 evidence，unclear 带 reason，absent 只存 value；逐项显式选择，不放入 modifiers |
| reason_code / reason | 常规 judged 不填写；异常先选 context_missing / ambiguous_intent / joke_or_serious / multi_function / boundary / other，可补一句；OTHER 必须补功能描述 |
| unit_objections[] | F 单元异议：unit_id、problem、proposal，不直接改冻结材料；无异议为 [] |
| legacy_projection（另存） | version、event_id、first_segment_id、label、reason；version 为 a2e-proj-v0（仅 v1.1 数据）或 a2e-proj-v1-draft（v1.2 草案，不作正式研究结果），label 为 A / B / C / D / E / U；记录所取首段及拒映原因，无可取首段为 null；格式例放在同名展示块，正式派生文件单独存，不写回原始标注 |

格式/文档与人工一致性是不同验收项；格式通过不代表标注正确或效果成立。

## 10. 场景覆盖矩阵

整理者在冻结前按现有 50 条填写事件×维度矩阵；同一事件可覆盖多个检查项。先查元数据，再看原文缺口，只用元数据和原文；CPED 原 DA/情绪仅用于筛查，不展示给 A/B。下面列出矩阵维度，实际覆盖须由材料核查填写，不在本指南预填覆盖结论。

| 维度 | 最小核查内容 |
|---|---|
| 交际场景 | 日常闲聊/信息交换、事务安排、问题求助、情绪表达/支持、请求邀请/承诺拒绝、冲突/边界、社交开闭场。 |
| 情绪 | 中性、积极、消极；原情绪标签用于来源分布检查，缺失也单列，不新增强度评分。 |
| 对话关系 | 优先检查不同说话人、角色组合和剧集；亲属/同事/朋友等只使用来源明确提供的信息，无依据记未知。 |
| 动作 | 11 个具体动作尽量各有至少 2 个、来自不同对话的判断机会；OTHER 不设配额，出现才分析。 |
| 表达方式与内容属性 | 8 项表达方式各尽量有明确正例和对照负例，evaluation 覆盖褒、贬、混合；self_disclosure 与 REQUEST.seek_care 另查支持数。少量 unclear 检验边界，不追求正负平衡；新增校准批 2 的覆盖单列，不宣称 8 个事件已覆盖全部边界。 |
| 回复结构 | 单动作、多动作有序回复、连续多消息、短应答，以及可能不可分的多功能片段。 |
| 语境难点 | 反问、间接请求、承诺与状态陈述、玩笑歧义、指代、更正/否定、上下文不足。 |
| 后续 F/G | F 有多信息点、无唯一首选；G 有合规候选及明显适用/不适用的判断机会，不能只收容易的正例。 |

矩阵逐事件记录各维度命中项、未知/缺失、基础样本中的支持数与缺口；整理者预测动作只用于选材，覆盖是否成立以独立标注结果为准。缺口用定向补缺事件填，最多 10 条，ID 从 `cped-s01` 起，标记 `sample_role=supplement`；基础 50 条标记 `sample_role=base`，不改动基础样本，补缺单独报告。

覆盖矩阵证明“边界被检查过”，不代表 CPED 自然分布；不要求穷举组合，也不把便利样本追称随机样本。

## 11. 旧 A–E 派生投影

### 11.1 a2e-proj-v0（仅适用于 v1.1 数据）

以下原文完整保留，输入特征定义及“无特征”仍按 v1.1 的四项判断，不用 v1.2 新字段重解释；原文提及的未来 v1 不等于下面尚未定稿的草案。

两套并存，M1 保持 A–E，不迁移、不重标；新协议原始标注不动，派生 A–E 另存。`legacy_projection_version = "a2e-proj-v0"`，桥接验证通过后升为 v1（`a2e-proj-v1`）。每条记录 version、所取 first_segment_id、label 与 reason（含拒映原因）；无法唯一确定时输出 `U`。规则要点如下：

- 跳过无特征的纯 ACK 开头
- 首段 seek_care→E，humor→D（同现记冲突，E 优先待验证）
- affection 单独不自动 E
- VALIDATE→A
- ASK/SUGGEST→B（针对前句的反问 A）
- INFORM/EVALUATE 直接接话 A、独立自述 C
- COMMIT/REFUSE 回应请求 A
- REQUEST/SOCIAL/OTHER 及多动作不一致→拒映。
- C≠F no_match。

### 11.2 a2e-proj-v1-draft（适用于 v1.2 的候选规则）

**研究侧（RC-PEM）2026-10-23 已核对，按下列规则定稿为候选投影；桥接验证通过前仍不与 v0 混报。**`legacy_projection_version = "a2e-proj-v1-draft"`。仅作候选派生展示，不代表桥接已通过或正式投影已启用；不能与 v0 混报。新协议原始标注、旧 A–E 原始结果与派生文件继续分开保存，不写回原始标注。记录 version、event_id、first_segment_id、label 与 reason；无可取首段时 first_segment_id=null，无法唯一确定时 label=U。

- 跳过“无特征的纯 ACK”开头：动作集合恰为 ACK，且 v1.2 的 8 项表达方式全为 absent，才允许跳过。evaluation 也必须为 absent；任一项 unclear 或未采集时不视为无特征。self_disclosure 属内容属性，不计入这 8 项。
- 首段 REQUEST 子类型 `seek_care`→E；不再读取旧 modifiers.seek_care。首段 humor=present→D；与 seek_care 同现记冲突，E 优先仍待验证。irony 不自动投 D。
- **E 候选（表白/示弱）**：首段为 INFORM 或 EVALUATE，且 rapport=present、emotion=present，内容是对对方的感情表达或示弱时，不投 A/C，输出 U，reason 写"E候选：表白/示弱"。用于在桥接验证中单独统计这部分损失。
- **D 候选（损/反话）**：首段 humor≠present，且 irony=present，或 evaluation=negative 且 rapport=present 时，不投 A，输出 U，reason 写"D候选：损/反话"。
- 冲突：seek_care 与 humor=present 同现、或 seek_care 与"D候选"同现，都记冲突。
- 旧 affection 条件改从 rapport 读取；rapport 单独出现不自动 E，不把一般亲和、礼貌、情绪或评价投为“撒娇”。
- VALIDATE→A。
- ASK/SUGGEST→B（针对前句的反问 A）。
- INFORM/EVALUATE 直接接话 A、独立自述 C；不能仅凭 self_disclosure 判 C。
- COMMIT/REFUSE 回应请求 A。
- 除上述 seek_care 或 humor 规则适用外，REQUEST/SOCIAL/OTHER 及多动作不一致→拒映；无法唯一确定为 U。
- C≠F no_match。桥接覆盖率、拒映原因和冲突单列，未验证前不宣称与 v0 等价。

## 12. 扩样（200 条）方案

**本指南不自动授权扩样，需负责人另行通知。**原 50 条、新增 8 条校准事件及定向补缺均属试标开发材料；200 条用于独立桥接验证。

| 项目 | 冻结方案 |
| --- | --- |
| 新事件与隔离 | 200 个新事件，按人物/源组与全部试标开发材料隔离。每条 1 份 12 动作标注 + 1 份直接 A–E 标注，两份由不同的人完成且互不可见。 |
| A–E 复标 | 60 条（30%），预先随机、按类分层选定；复标者不看任何模型结果。 |
| 12 动作复标 | 40 条（20%），预先选定；复标者不看任何模型结果。 |
| 类别覆盖 | 每个 A–E 类在 200 条中各 ≥20 条；不足时分层补抽并单列。抽样时用 CPED 情绪/DA 粗筛保证 D（调侃）、E（撒娇）够数。 |
| 桥接合格线 | 在全部 200 条上计算：覆盖率（非 U）≥90%、桥接 κ ≥.60、宏 F1 ≥.80、每类 F1 ≥.70；U 作为拒映保留在全量分母，不只报可映射子集：计算宏 F1 和每类 F1 时，U 一律算错（计为真实类别的漏召回）；桥接 κ 只在可映射子集上计算，同时报覆盖率。 |
| A–E 人际一致性 | 在 60 条复标上报 κ 点估计（≥.60）与 95% 区间。全部合格线满足才把投影作为辅助桥接，否则两套分开报告。 |
| LLM 建议另轨 | 人工先独立提交，之后才可显示；LLM 不算独立标注者，不进任何一致性。另轨存放建议，并抽查人机一致项。 |
| 12 动作主标 | 200 份，约 8 h。 |
| 直接 A–E 主标 | 200 份，约 2.5 h。 |
| A–E 复标工时 | 60 份，约 0.75 h。 |
| 12 动作复标工时 | 40 份，约 1.7 h。 |
| 合计 | 约 500 份、约 13 h，两人分摊每人约 6.5 h；为估算，按实测调整。 |
