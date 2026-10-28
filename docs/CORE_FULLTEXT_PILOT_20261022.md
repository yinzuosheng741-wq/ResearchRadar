# 核心全文试点覆盖与质量台账

日期：2026-10-22。20 篇定向样本；每篇检查 1 个渲染页面，未完成全文语义认证。标签基于标题和抽样页，不代表相应任务均有完整答案。

2026-10-28 增量：针对 GLORIA 与 LAGOS-US 的指定原页完成第一组任务级标注与冻结一致性校验，见[证据任务试点](EVIDENCE_PILOT_20261028.md)。它不将本台账中的任何全文自动改为语义认证通过。

规范语料与 FAISS 的 ID、正文载荷、页码、数量核对通过；不证明向量数值与文本对应或召回效果。

| 论文 | 任务方向 | 抽查 PDF 页 | 观察与限制 |
| --- | --- | --- | --- |
| LAGOS-US LANDSAT: Remotely sensed water quality estimates for U.S. lakes over 4 ha from 1984 to 2020 | 长时间序列湖泊产品 | 2 | 单栏正文及大型对照表；表头有合并列，线性文本不保留网格关系。 |
| GLORIA - A globally representative hyperspectral in situ dataset for optical sensing of water quality | 现场光谱与质量控制 | 8 | 质量标记表有长描述、多行单元格，不能仅凭段落顺序关联数值。 |
| A bio-optical database for the remote sensing of water quality in BRAZil coAstal and inland waters (BRAZA) | 现场生物光学数据 | 2 | 背景页为单栏密集正文；页面可读，不代表数据文件已导入。 |
| A Chlorophyll-a Algorithm for Landsat-8 Based on Mixture Density Networks | 机器学习反演 | 6 | 双栏正文、模型输入表和公式共存；提取文本有字间空格异常。 |
| Remote sensing of the chlorophyll-a based on OLI/Landsat-8 and MSI/Sentinel-2A (Barra Bonita reservoir, Brazil) | 传感器比较 | 11 | 双栏正文、空间分布图与直方图；图像本身未作为结构化数值证据。 |
| Chlorophyll-a Detection Algorithms at Different Depths Using In Situ, Meteorological, and Remote Sensing Data in a Chilean Lake | 深度与实测条件 | 10 | 训练测试样本表和多子图；抽样页能看到分层条件，需保留深度限定。 |
| Water Quality Retrieval from Landsat-9 (OLI-2) Imagery and Comparison to Sentinel-2 | 跨传感器验证 | 3 | 双栏正文与研究区地图；PDF 页码与期刊排版编号需要区分。 |
| Mapping of total suspended solids using Landsat imagery and machine learning | 机器学习反演 | 7 | 双栏方法与结果文本；数值与各实验数据子集关联待逐项标注。 |
| Robust algorithm for estimating total suspended solids (TSS) in inland and nearshore coastal waters | 半分析反演 | 30 | 带行号的稿件版式；跨行与行号可能污染检索，抽样页无真实表格。 |
| Low-Cost Sensors and Multitemporal Remote Sensing for Operational Turbidity Monitoring in an East African Wetland Environment | 传感器与现场验证 | 2 | 双栏方法与研究区描述；开头承接上一页，跨页段落需要回查。 |
| Advantages of high quality SWIR bands for ocean colour processing: Examples from Landsat-8 | 大气校正 | 2 | 双栏正文、公式与底部宽表；提取的公式出现非原始算符字符，公式使用受限。 |
| Optical types of inland and coastal waters | 水体分类 | 3 | 双栏综述正文，自动表格标记是引用提及；不能当成实际表格页。 |
| Water Constituents and Water Depth Retrieval from Sentinel-2A—A First Evaluation in an Oligotrophic Lake | 水深与组分联合反演 | 14 | 多幅空间图与讨论正文；图中像元信息不在文本证据中。 |
| Use of Sentinel 2 – MSI for water quality monitoring at Alqueva reservoir, Portugal | 水质监测 | 2 | 双栏正文包含波段表、公式和方法标题；需原图核对公式算符。 |
| Comparison of Eulerian and Lagrangian transport models for harmful algal bloom forecasts in Lake Erie | 输运预测 | 8 | 带行号的正文和复杂表头；年份与云覆盖类别应整体引用，不能拆开推断。 |
| Improved seasonal prediction of harmful algal blooms in Lake Erie using large-scale climate indices | 季节预测 | 3 | 双栏正文与横跨两栏的性能表；模型公式和对应年份的关系待结构化审核。 |
| New capabilities of Sentinel-2A/B satellites combined with in situ data for monitoring small harmful algal blooms in complex coastal waters | 藻华检测 | 8 | 卫星复合图与多栏讨论；图注和图面都有信息，文本检索不能替代图面阅读。 |
| A review of remote sensing-based water quality monitoring in turbid coastal waters | 领域综述 | 7 | 双栏正文、反演公式与方法小节；文献综述结论不等于原始实验验证。 |
| Living up to the Hype of Hyperspectral Aquatic Remote Sensing: Science, Resources and Outlook | 高光谱与大气校正综述 | 9 | 双栏正文及横跨两栏的软件对照表；工具网址与描述需保持行对应。 |
| Imaging Spectrometry of Inland and Coastal Waters: State of the Art, Achievements and Perspectives | 成像光谱综述 | 18 | 单栏密集正文，所示页涉及底栖应用；不能把论文所有页面都标为水质证据。 |

## 后续缺口

- 所有全文块缺少章节字段；先针对核心证据补章节定位，不使用未经核验的自动标签宣布完成。
- 现有线性文本未保存表格网格、图像像元与公式版式；比较数值和公式应回查原页。
- 样本包含 3 篇 license 为空和 3 篇 other-oa；不据此推断可再分发全文。
- 当前尚无按研究任务标注的完整覆盖率；新建问题草案待核验后再冻结集合。
- 完整机器报告见 artifacts/fulltext_quality_20261022_final.json；逐篇审查记录见 artifacts/core_fulltext_review_20261022.json。
- 早期 fulltext_quality_20261022.json 使用 page_content 比较富化文本，产生假阳性；该草稿已撤去。v2 为修正载荷读取后的中间报告，最终以 final 报告为准。
