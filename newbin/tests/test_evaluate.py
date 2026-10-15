#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NewBin 基准评测模块单元测试 (test_evaluate.py)
"""

import os
import unittest
import tempfile
import pandas as pd
from newbin.evaluate import (
    load_fasta_lengths,
    load_clustering_results,
    load_ground_truth,
    compute_benchmark_metrics,
    format_benchmark_report,
    run_benchmark
)


class TestEvaluateModule(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_compute_benchmark_metrics_perfect(self):
        """测试完美分箱场景：完全召回、无污染"""
        contig2bin = {
            'c1': 'bin_1', 'c2': 'bin_1',
            'c3': 'bin_2', 'c4': 'bin_2',
        }
        contig2genome = {
            'c1': 'G1', 'c2': 'G1',
            'c3': 'G2', 'c4': 'G2',
        }
        lengths = {'c1': 1000, 'c2': 2000, 'c3': 3000, 'c4': 4000}

        results = compute_benchmark_metrics(contig2bin, contig2genome, lengths)
        summary = results["summary"]
        details = results["genome_details"]

        self.assertEqual(summary["Total_Genomes"], 2)
        self.assertEqual(summary["Total_Bins"], 2)
        self.assertEqual(summary["High_Quality_MAGs (HQ)"], 2)
        self.assertEqual(summary["Medium_Quality_MAGs (MQ)"], 0)
        self.assertAlmostEqual(summary["Mean_Completeness"], 1.0)
        self.assertAlmostEqual(summary["Mean_Contamination"], 0.0)
        self.assertAlmostEqual(summary["Mean_F1_Score"], 1.0)
        self.assertAlmostEqual(summary["ARI"], 1.0)
        self.assertAlmostEqual(summary["NMI"], 1.0)

    def test_compute_benchmark_metrics_mixed_and_tiers(self):
        """测试存在污染、拆分及不同质量等级评定"""
        # G1: c1 (900bp), c2 (100bp) -> 总长 1000bp
        # G2: c3 (2000bp), c4 (1000bp) -> 总长 3000bp
        # G3: c5 (1000bp) -> 总长 1000bp
        #
        # Bin_A: c1 (900bp of G1) + c4 (1000bp of G2) -> 总长 1900bp
        #   对于 G1: Comp = 900/1000 = 90%, Purity = 900/1900 = 47.37%, Contam = 1 - Purity = 52.63% -> Over-Contaminated
        # Bin_B: c3 (2000bp of G2) -> 总长 2000bp
        #   对于 G2: Comp = 2000/3000 = 66.67%, Contam = 0.0% -> MQ (Comp >= 50% & Contam < 10%)
        # Bin_C: c5 (400bp of G3, 假设切碎) ...
        contig2bin = {
            'c1': 'Bin_A',
            'c4': 'Bin_A',
            'c3': 'Bin_B',
            'c2': 'Bin_C',
        }
        contig2genome = {
            'c1': 'G1',
            'c2': 'G1',
            'c3': 'G2',
            'c4': 'G2',
            'c5': 'G3', # G3 完全没有被 binned
        }
        lengths = {
            'c1': 900,
            'c2': 100,
            'c3': 2000,
            'c4': 1000,
            'c5': 1000,
        }

        results = compute_benchmark_metrics(contig2bin, contig2genome, lengths)
        details = results["genome_details"].set_index("Genome")

        # G1 匹配 Bin_A
        self.assertEqual(details.loc['G1', 'Best_Bin'], 'Bin_A')
        self.assertAlmostEqual(details.loc['G1', 'Completeness'], 0.90)
        self.assertAlmostEqual(details.loc['G1', 'Purity'], 900 / 1900)
        self.assertEqual(details.loc['G1', 'Tier'], 'Over-Contaminated')

        # G2 匹配 Bin_B
        self.assertEqual(details.loc['G2', 'Best_Bin'], 'Bin_B')
        self.assertAlmostEqual(details.loc['G2', 'Completeness'], 2000 / 3000)
        self.assertAlmostEqual(details.loc['G2', 'Contamination'], 0.0)
        self.assertEqual(details.loc['G2', 'Tier'], 'MQ')

        # G3 未分箱
        self.assertEqual(details.loc['G3', 'Best_Bin'], 'Unbinned')
        self.assertAlmostEqual(details.loc['G3', 'Completeness'], 0.0)
        self.assertEqual(details.loc['G3', 'Tier'], 'LQ')

    def test_file_loading_and_pipeline(self):
        """测试从各类文件加载并运行端到端评测流水线"""
        # 创建 FASTA
        fasta_path = os.path.join(self.temp_dir.name, "test.fasta")
        with open(fasta_path, "w") as f:
            f.write(">ctg1\nACGTACGT\n>ctg2\nACGTACGTACGT\n>ctg3\nACGT\n")

        # 验证读取长度: ctg1=8, ctg2=12, ctg3=4
        lens = load_fasta_lengths(fasta_path)
        self.assertEqual(lens['ctg1'], 8)
        self.assertEqual(lens['ctg2'], 12)
        self.assertEqual(lens['ctg3'], 4)

        # 创建真值 CSV
        gt_path = os.path.join(self.temp_dir.name, "gt.csv")
        pd.DataFrame({
            "ctg_id": ["ctg1", "ctg2", "ctg3"],
            "clr": ["Genome_A", "Genome_A", "Genome_B"]
        }).to_csv(gt_path, index=False)

        # 创建预测分箱 CSV
        bin_path = os.path.join(self.temp_dir.name, "clusters.csv")
        pd.DataFrame({
            "sequence": ["ctg1", "ctg2", "ctg3"],
            "cluster": ["bin_1", "bin_1", "bin_2"]
        }).to_csv(bin_path, index=False)

        # 运行 run_benchmark
        out_dir = os.path.join(self.temp_dir.name, "eval_out")
        res = run_benchmark(
            binned_path=bin_path,
            gt_path=gt_path,
            fasta_path=fasta_path,
            output_dir=out_dir
        )

        self.assertEqual(res["summary"]["High_Quality_MAGs (HQ)"], 2)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "evaluation_summary.tsv")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "evaluation_details.tsv")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "evaluation_strata.tsv")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "evaluation_report.txt")))

    def test_short_contig_rescue_metrics(self):
        """测试 <2000bp 短序列与质粒拯救能力计算"""
        # G1:
        #   long_1: 10000 bp -> bin_1
        #   short_1: 1500 bp -> bin_1 (成功拯救)
        #   short_2: 1200 bp -> bin_2 (误分类)
        #   short_3: 800 bp -> 未分箱
        # G2:
        #   long_2: 8000 bp -> bin_2
        #   short_4: 1800 bp -> bin_2 (成功拯救)
        contig2bin = {
            'long_1': 'bin_1',
            'short_1': 'bin_1',
            'short_2': 'bin_2',
            # short_3 未分箱
            'long_2': 'bin_2',
            'short_4': 'bin_2',
        }
        contig2genome = {
            'long_1': 'G1',
            'short_1': 'G1',
            'short_2': 'G1',
            'short_3': 'G1',
            'long_2': 'G2',
            'short_4': 'G2',
        }
        lengths = {
            'long_1': 10000,
            'short_1': 1500,
            'short_2': 1200,
            'short_3': 800,
            'long_2': 8000,
            'short_4': 1800,
        }

        results = compute_benchmark_metrics(contig2bin, contig2genome, lengths, short_threshold=2000)
        summary = results["summary"]
        strata = results["strata_details"]

        # 短序列 (<2000 bp) 共有 4 条: short_1, short_2, short_3, short_4
        # 总条数 = 4
        # 总长度 = 1500 + 1200 + 800 + 1800 = 5300 bp
        # 成功拯救的条数: short_1(G1->bin_1) 和 short_4(G2->bin_2) 共 2 条
        # 成功拯救的长度: 1500 + 1800 = 3300 bp
        self.assertEqual(summary["Total_Short_Contigs"], 4)
        self.assertEqual(summary["Rescued_Short_Contigs"], 2)
        self.assertAlmostEqual(summary["Short_Rescue_Rate_Count"], 2 / 4) # 50%
        self.assertAlmostEqual(summary["Short_Rescue_Rate_BP"], 3300 / 5300)

        # 检查分层统计 df
        short_row = strata.iloc[0]
        self.assertEqual(short_row["Total_Contigs"], 4)
        self.assertEqual(short_row["Rescued_Contigs"], 2)
        self.assertEqual(short_row["Misclassified_Contigs"], 1) # short_2
        self.assertEqual(short_row["Unbinned_Contigs"], 1)      # short_3


if __name__ == "__main__":
    unittest.main()

