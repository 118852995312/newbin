import unittest
import os
import sys
import numpy as np
import pandas as pd
from io import StringIO
from pandas.testing import assert_frame_equal

# 将项目目录加入 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from newbin.generate_kmer import generate_feature_mapping, generate_kmer_features_from_fasta
from newbin.generate_coverage import calculate_coverage, combine_cov


class TestKmerAndCoverage(unittest.TestCase):
    def setUp(self):
        # 基础测试 FASTA 数据
        self.fasta_content = (
            '>example\n'
            'GCATGAGACAGTGTTTGCTGCAATACTCCCTGAGGCGCTTGCATATTATGAGAAGAATGCTCCCAAGGGAGAGTGCGTGATCGTTATAGAGGGAAAGAGC'
            'AGGCTTGAGATTCGGGAAGAAGAGAAAGCACAGTGGGAACAGCTTACCATTGAAGAACATATGGAACGCTATCTCTCCGGCGGTATGGAGAAAAAGGAAG'
            'CTATGAAGCGGGTGGCAAAGGATCGGGGCGTAAGTAAAAGAGATATTTATCAAGCGCTTCTTTAAACCTTCATATGAATGAAATAAAAACAGGTATCGGG'
            'TGTGTTC\n'
            '>contig_with_n\n'
            'AAAAAAAAAAAAAAANAAAAAAAAAAAA\n'
            '>short_contig\n'
            'ATGC\n'
        )

    def test_feature_mapping_4mer(self):
        """测试 4-mer canonical 特征维度为 136"""
        fm, num_features = generate_feature_mapping(4)
        self.assertEqual(num_features, 136)
        self.assertEqual(len(fm), 256)

    def test_kmer_calculation_basic(self):
        """测试 k-mer 频率计算与归一化"""
        df = generate_kmer_features_from_fasta(
            StringIO(self.fasta_content), length_threshold=10, kmer_len=4)
        self.assertEqual(df.shape, (2, 136))
        # 验证行和为 1.0 (归一化分布)
        np.testing.assert_allclose(df.sum(axis=1).values, [1.0, 1.0], rtol=1e-5)

    def test_kmer_whitelist(self):
        """测试 contig_whitelist 白名单预筛选"""
        df = generate_kmer_features_from_fasta(
            StringIO(self.fasta_content),
            length_threshold=10,
            kmer_len=4,
            contig_whitelist={'example'}
        )
        self.assertEqual(list(df.index), ['example'])
        self.assertEqual(df.shape, (1, 136))

    def test_coverage_calculation(self):
        """测试 coverage 覆盖度提取 mean 和 var"""
        depth_stream = [
            'example\t0\t500\t10\n',
            'example\t500\t1500\t20\n',
            'contig_2\t0\t1200\t5\n'
        ]

        cov = calculate_coverage(
            depth_stream, 'sample.bam', is_combined=True
        )
        self.assertIsNotNone(cov)
        self.assertIn('sample.bam_mean', cov.columns)
        self.assertIn('sample.bam_var', cov.columns)
        self.assertEqual(len(cov), 2)

    def test_combine_cov_level_shape_id(self):
        """测试结合文档中的 4 样本 toy example 验证 level、shape、log_id 特征"""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            bams = ['S1.bam', 'S2.bam', 'S3.bam', 'S4.bam']
            # Contig_i: [10, 50, 30, 5]
            # Contig_k (3x copy number): [30, 150, 90, 15]
            mu_i = [10.0, 50.0, 30.0, 5.0]
            mu_k = [30.0, 150.0, 90.0, 15.0]
            vars_val = [1.0, 2.0, 1.5, 0.5]

            for idx, bam in enumerate(bams):
                df = pd.DataFrame({
                    f'{bam}_mean': [mu_i[idx], mu_k[idx]],
                    f'{bam}_var': [vars_val[idx], vars_val[idx] * 3]
                }, index=['contig_i', 'contig_k'])
                df.to_csv(os.path.join(tmpdir, f'{bam}_{idx}_data_cov.csv'))

            data_cov = combine_cov(tmpdir, bams, eps=0.0)

            # 1. 验证维度为 2M + 1 = 9
            self.assertEqual(data_cov.shape, (2, 9))
            self.assertIn('cov_level', data_cov.columns)

            # 2. 验证 level: ln(3) 差值进入 level
            level_i = data_cov.loc['contig_i', 'cov_level']
            level_k = data_cov.loc['contig_k', 'cov_level']
            expected_level_i = np.mean(np.log(mu_i))
            expected_level_k = np.mean(np.log(mu_k))
            self.assertAlmostEqual(level_i, expected_level_i, places=4)
            self.assertAlmostEqual(level_k, expected_level_k, places=4)
            self.assertAlmostEqual(level_k - level_i, np.log(3.0), places=4)

            # 3. 验证 shape: 3倍重复区的 shape 与原始 contig 严格相同且和为 0
            shape_cols = [c for c in data_cov.columns if c.startswith('cov_shape_')]
            self.assertEqual(len(shape_cols), 4)
            shape_i = data_cov.loc['contig_i', shape_cols].values.astype(float)
            shape_k = data_cov.loc['contig_k', shape_cols].values.astype(float)
            np.testing.assert_allclose(shape_i, shape_k, atol=1e-5)
            self.assertAlmostEqual(shape_i.sum(), 0.0, places=5)
            self.assertAlmostEqual(shape_k.sum(), 0.0, places=5)

    def test_real_fasta_kmer_extraction(self):
        """测试使用本地实际 FASTA 数据提取 136 维 k-mer"""
        fasta_path = os.path.join(os.path.dirname(__file__), 'test_data', 'contigs.fasta')
        if os.path.exists(fasta_path):
            df = generate_kmer_features_from_fasta(fasta_path, length_threshold=1000, kmer_len=4)
            self.assertGreater(len(df), 0)
            self.assertEqual(df.shape[1], 136)
            np.testing.assert_allclose(df.sum(axis=1).values, np.ones(len(df)), rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
