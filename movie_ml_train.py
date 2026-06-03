"""
附加题3: 前沿话题 —— 基于 K8s 的分布式机器学习 (Spark MLlib)
学号：2023112584  姓名：吕昊阳
学号：2023112474  姓名：范原琿

前沿方向: 云原生分布式 AI 训练
技术栈: Spark MLlib on Kubernetes (Spark Operator)
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year as spark_year, when, isnan, isnull
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder, StandardScaler
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import time

spark = SparkSession.builder \
    .appName("DistributedMLTraining") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

print("=" * 60)
print("附加题3: 云原生分布式机器学习 — Spark MLlib on Kubernetes")
print("=" * 60)

# 1. 加载数据
print("\n[1] 加载数据...")
df = spark.read \
    .option("multiLine", "true") \
    .option("escape", "\"") \
    .csv("/data/douban_movies.csv", header=True, inferSchema=True, encoding="UTF-8")

print(f"    原始行数: {df.count()}")
print(f"    Executor 数: {spark.sparkContext._conf.get('spark.executor.instances', 'default')}")

# 2. 数据预处理
print("\n[2] 数据预处理 (Spark ML Pipeline)...")

df_clean = df.select("title", "rating_score", "rating_count", "year", "genres") \
    .filter(col("rating_score").isNotNull()) \
    .filter(col("rating_count").isNotNull()) \
    .filter(col("year").isNotNull()) \
    .filter(col("year") > 1900) \
    .filter(col("genres").isNotNull())

# 特征工程: 年份转换、评论数取对数
df_feat = df_clean \
    .withColumn("year_norm", (col("year") - 1950) / 100) \
    .withColumn("log_rating_count", when(col("rating_count") > 0,
        col("rating_count")).otherwise(1).cast("double"))

print(f"    有效样本数: {df_feat.count()}")

# 3. Train/Test 划分 (80/20)
train_df, test_df = df_feat.randomSplit([0.8, 0.2], seed=42)
print(f"\n[3] 数据划分: Train={train_df.count()}, Test={test_df.count()}")

# 4. 构建 ML Pipeline
print("\n[4] 构建 ML Pipeline...")

# 特征向量: year_norm + log_rating_count
assembler = VectorAssembler(
    inputCols=["year_norm", "log_rating_count"],
    outputCol="features"
)

scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withStd=True,
    withMean=True
)

# 模型: 随机森林回归 + 线性回归 (对比)
rf = RandomForestRegressor(
    featuresCol="scaled_features",
    labelCol="rating_score",
    numTrees=50,
    maxDepth=10,
    seed=42
)

lr = LinearRegression(
    featuresCol="scaled_features",
    labelCol="rating_score",
    maxIter=50,
    regParam=0.01
)

# ============================================================
# 5. 训练随机森林模型
# ============================================================
print("\n[5.1] 训练 RandomForestRegressor (分布式)...")
print("      模型: RandomForest (50 trees, maxDepth=10)")
print("      训练节点: driver + executors (K8s Pods)")

pipeline_rf = Pipeline(stages=[assembler, scaler, rf])

start = time.time()
model_rf = pipeline_rf.fit(train_df)
train_time_rf = time.time() - start
print(f"      训练耗时: {train_time_rf:.2f}s")

# 评估
pred_rf = model_rf.transform(test_df)
evaluator_rmse = RegressionEvaluator(labelCol="rating_score", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="rating_score", predictionCol="prediction", metricName="r2")
evaluator_mae = RegressionEvaluator(labelCol="rating_score", predictionCol="prediction", metricName="mae")

rmse_rf = evaluator_rmse.evaluate(pred_rf)
r2_rf = evaluator_r2.evaluate(pred_rf)
mae_rf = evaluator_mae.evaluate(pred_rf)

print(f"      RMSE: {rmse_rf:.4f}")
print(f"      R²:   {r2_rf:.4f}")
print(f"      MAE:  {mae_rf:.4f}")

# ============================================================
# 6. 训练线性回归模型 (对比)
# ============================================================
print("\n[5.2] 训练 LinearRegression (对比)...")

pipeline_lr = Pipeline(stages=[assembler, scaler, lr])

start = time.time()
model_lr = pipeline_lr.fit(train_df)
train_time_lr = time.time() - start
print(f"      训练耗时: {train_time_lr:.2f}s")

pred_lr = model_lr.transform(test_df)
rmse_lr = evaluator_rmse.evaluate(pred_lr)
r2_lr = evaluator_r2.evaluate(pred_lr)
mae_lr = evaluator_mae.evaluate(pred_lr)

print(f"      RMSE: {rmse_lr:.4f}")
print(f"      R²:   {r2_lr:.4f}")
print(f"      MAE:  {mae_lr:.4f}")

# ============================================================
# 7. 结果汇总
# ============================================================
print("\n" + "=" * 60)
print("模型对比:")
print("-" * 40)
print(f"{'指标':<10} {'RandomForest':<20} {'LinearRegression':<20}")
print(f"{'RMSE':<10} {rmse_rf:<20.4f} {rmse_lr:<20.4f}")
print(f"{'R²':<10}   {r2_rf:<20.4f} {r2_lr:<20.4f}")
print(f"{'MAE':<10}  {mae_rf:<20.4f} {mae_lr:<20.4f}")
print(f"{'训练时间':<10} {train_time_rf:<20.2f}s {train_time_lr:<20.2f}s")

print("\n前沿技术分析:")
print("  1. 云原生架构: 训练任务运行于 K8s 集群，利用 Spark Operator 实现")
print("     自动化的分布式资源调度，无需手动管理计算节点。")
print("  2. 弹性伸缩: 可根据数据规模动态调整 Executor 数量，实现")
print("     训练成本与效率的平衡。")
print("  3. 分布式训练: Spark MLlib 的 RandomForest 天然支持分布式并行")
print("     每棵树可在不同 Executor 上独立训练，线性加速。")
print("  4. 与传统方案对比: 相比单机 Scikit-Learn，本方案可处理 TB 级")
print("     数据集，且与 K8s 生态无缝集成（监控、日志、CI/CD）。")

print("\n相关资料:")
print("  - KubeFlow: https://www.kubeflow.org/")
print("  - Spark Operator: https://github.com/kubeflow/spark-operator")
print("  - Spark MLlib: https://spark.apache.org/mllib/")
print("=" * 60)

spark.stop()
