import unittest
import os
import sys
import tempfile
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

    def test_self_sample_coverage_m_less_than_5(self):
        """测试 M < 5 (如单样本 M=1) 时，由自身样本 BAM 计算得到 3 维 Coverage 特征或 11 维对齐特征"""
        with tempfile.TemporaryDirectory() as tmpdir:
            bams = ['self_sample']
            mu_val = [45.0]
            var_val = [3.2]

            df = pd.DataFrame({
                f'{bams[0]}_mean': [mu_val[0], mu_val[0] * 2],
                f'{bams[0]}_var': [var_val[0], var_val[0] * 4]
            }, index=['contig_1', 'contig_2'])
            df.to_csv(os.path.join(tmpdir, f'{bams[0]}_0_data_cov.csv'))

            # 1. 默认 pad_to_max=False: 输出自然 2M + 1 = 3 维特征
            data_cov_natural = combine_cov(tmpdir, bams, eps=1e-5, pad_to_max=False)
            self.assertEqual(data_cov_natural.shape, (2, 3))
            self.assertIn('cov_level', data_cov_natural.columns)
            self.assertIn('cov_shape_self_sample', data_cov_natural.columns)
            self.assertIn('cov_log_id_self_sample', data_cov_natural.columns)

            # 单样本 shape 相对自身为 0
            np.testing.assert_allclose(data_cov_natural['cov_shape_self_sample'].values, [0.0, 0.0], atol=1e-5)

            # 2. pad_to_max=True: 输出对齐固定架构的 11 维特征
            data_cov_padded = combine_cov(tmpdir, bams, eps=1e-5, pad_to_max=True)
            self.assertEqual(data_cov_padded.shape, (2, 11))
            self.assertIn('cov_shape_pad_1', data_cov_padded.columns)
            self.assertIn('cov_log_id_pad_1', data_cov_padded.columns)
            # 真实特征保持一致
            self.assertAlmostEqual(data_cov_padded.loc['contig_1', 'cov_level'],
                                   data_cov_natural.loc['contig_1', 'cov_level'])

    def test_multi_sample_self_sample_coverage_matching(self):
        """测试多样本模式下 M < 5 时，各样本正确使用自己专属 BAM 计算 Coverage 并通过 split_data 切分"""
        from newbin.utils import split_data

        with tempfile.TemporaryDirectory() as tmpdir:
            sample_list = ['S1', 'S2']
            bams = ['/path/to/S1.sorted.bam', '/path/to/S2.sorted.bam']

            # 模拟 S1 和 S2 分别在自己 BAM 中的 coverage 数据
            # S1 contigs: S1:ctg1, S1:ctg2
            # S2 contigs: S2:ctg1, S2:ctg2
            df_s1 = pd.DataFrame({
                'S1.sorted.bam_mean': [10.0, 20.0],
                'S1.sorted.bam_var': [2.0, 4.0]
            }, index=['S1:ctg1', 'S1:ctg2'])
            df_s1.to_csv(os.path.join(tmpdir, 'S1.sorted.bam_0_data_cov.csv'))

            df_s2 = pd.DataFrame({
                'S2.sorted.bam_mean': [100.0, 200.0],
                'S2.sorted.bam_var': [20.0, 40.0]
            }, index=['S2:ctg1', 'S2:ctg2'])
            df_s2.to_csv(os.path.join(tmpdir, 'S2.sorted.bam_1_data_cov.csv'))

            # 对 S1 匹配 index 0
            cov_s1 = combine_cov(tmpdir, ['/path/to/S1.sorted.bam'], is_combined=False, bam_indices=[0])
            cov_s1 = cov_s1.reset_index()
            cov_s1.columns = ['contig_name'] + list(cov_s1.columns[1:])
            part_s1 = split_data(cov_s1, 'S1', separator=':', is_combined=False)

            self.assertEqual(len(part_s1), 2)
            self.assertIn('ctg1', part_s1.index)
            self.assertIn('ctg2', part_s1.index)
            self.assertEqual(part_s1.shape[1], 3)  # level, shape, log_id

            # 对 S2 匹配 index 1
            cov_s2 = combine_cov(tmpdir, ['/path/to/S2.sorted.bam'], is_combined=False, bam_indices=[1])
            cov_s2 = cov_s2.reset_index()
            cov_s2.columns = ['contig_name'] + list(cov_s2.columns[1:])
            part_s2 = split_data(cov_s2, 'S2', separator=':', is_combined=False)

            self.assertEqual(len(part_s2), 2)
            self.assertIn('ctg1', part_s2.index)
            self.assertIn('ctg2', part_s2.index)
            self.assertGreater(part_s2.loc['ctg1', 'cov_level'], part_s1.loc['ctg1', 'cov_level'])


if __name__ == '__main__':
    unittest.main()


