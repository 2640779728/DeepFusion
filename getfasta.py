import csv
import os
from Bio import SeqIO

def load_genome_sequences(genome):
    genome_sequences = {}
    genome_dir = f'data/{genome}'
    try:
        for filename in os.listdir(genome_dir):
            if filename.endswith('.fa'):
                path = os.path.join(genome_dir, filename)
                with open(path, 'r') as fasta_file:
                    for record in SeqIO.parse(fasta_file, 'fasta'):
                        genome_sequences[record.id] = str(record.seq)
    except FileNotFoundError:
        print(f'Genome directory {genome_dir} not found.')
    except Exception as e:
        print(f'Error loading genome sequences: {e}')
    return genome_sequences

def generate_fasta_file(csv_file, fasta_file, genome, cell_name):
    print('正在生成fasta文件...')
    sequences = load_genome_sequences(genome)
    count = 0  # 计数器

    output_directory = f'data/{cell_name}'
    os.makedirs(output_directory, exist_ok=True)

    try:
        with open(csv_file, 'r') as file, open(fasta_file, 'w') as output_file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                chromosome, start, end = row['chrom'], int(row['start']), int(row['end'])
                sequence = sequences.get(chromosome, '')[start-1:end]
                if sequence:
                    output_file.write(f'>{chromosome}\n{sequence}\n')
                    count += 1
                    if count % 100 == 0:
                        print(f'已写入 {count} 个序列...')
    except FileNotFoundError:
        print(f'Error: {csv_file} not found.')
    except Exception as e:
        print(f'Unexpected error: {e}')

    print(f'生成的fasta文件已保存在 {fasta_file} 中.')

if __name__ == '__main__':
    import sys
    try:
        cell_name = str(sys.argv[1])
        genome = str(sys.argv[2])
        csv_file = f'data/{cell_name}/{cell_name}.csv'
        fasta_file = f'data/{cell_name}/{cell_name}.fasta'
        generate_fasta_file(csv_file, fasta_file, genome, cell_name)
    except IndexError:
        print("Usage: script.py [cell_name] [genome]")
