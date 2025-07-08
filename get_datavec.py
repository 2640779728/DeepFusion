import os
import sys
from multiprocessing import Process, Lock
import Bio
import gensim
import numpy as np
import pandas as pd
from Bio import SeqIO

NULL_VEC = np.zeros((100))

def get_kmer(dnaSeq, K):
    dnaSeq = dnaSeq.upper()
    l = len(dnaSeq)
    return [dnaSeq[i:i+K] for i in range(l-K+1)]

def seq_to_vec(cell_name, seq_records, embedding_matrix, K, lock):
    print('Process %d is running' % K)
    code_file = f'./data/{cell_name}/{K}mer_datavec.csv'
    seqid = 1
    for seq_record in seq_records:
        dnaSeq = str(seq_record.seq)
        kmers = get_kmer(dnaSeq, K)
        code = [embedding_matrix[kmer] if kmer in embedding_matrix else NULL_VEC for kmer in kmers]
        ave = np.mean(code, axis=0)
        id_and_ave = pd.DataFrame([[seqid] + ave.tolist()], columns=['id'] + [f'feature_{i}' for i in range(ave.shape[0])])
        with lock:
            # 无论何时都不添加列标签
            id_and_ave.to_csv(code_file, mode='a', header=False, index=False)
        print(f'The {seqid}th sequence is done')
        seqid += 1

if __name__ == '__main__':
    cell_name = str(sys.argv[1])
    genome = str(sys.argv[2])
    seq_file = f'./data/{cell_name}/{cell_name}.fasta'
    seq_records = list(Bio.SeqIO.parse(seq_file, 'fasta'))
    embedding_matrix = gensim.models.KeyedVectors.load_word2vec_format(f"./data/{genome}_embedding.w2v", binary=False)
    print(f'Total records: {len(seq_records)}')
    print(f'Main process {os.getpid()} is running...')
    lock = Lock()
    processes = []
    for K in range(4, 7):
        p = Process(target=seq_to_vec, args=(cell_name, seq_records, embedding_matrix, K, lock))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    print(f'Main process {os.getpid()} is done...')
