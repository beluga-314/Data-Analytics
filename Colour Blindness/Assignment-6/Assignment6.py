import numpy as np
import time
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.special import softmax

class GeneticAnalysis:

    def __init__(self, data_dir='../data'):
        self.data_dir = data_dir
        self.bwt, self.Ranks = self.load_data(os.path.join(data_dir, 'chrX_last_col.txt'), ranks=True)
        self.indices = self.load_data(os.path.join(data_dir, 'chrX_map.txt'))
        self.ref = self.load_data(os.path.join(data_dir, 'chrX.fa'))[1:]
        # Initialize counts for match in each exon of Red and Green genes
        self.RExons = np.zeros(6)
        self.GExons = np.zeros(6)

    def load_data(self, data_dir, ranks=False):
        if ranks:
            lines = []
            Ranks = {'A': [], 'C': [], 'G': [], 'T': []}
            with open(data_dir, 'r') as f:
                prevcounts = [0, 0, 0, 0]
                for line in f:
                    lines.append(line)
                    for i, s in enumerate(Ranks.keys()):
                        currcount = line.count(s) + prevcounts[i]
                        Ranks[s].append(currcount)
                        prevcounts[i] = currcount
            return lines, Ranks
        else:
            f = open(data_dir, 'r')
            lines = f.readlines()
            f.close()
            return lines

    def Match(self, str1, str2):
        mismatch = 0
        for i in range(len(str2)):
            if str1[i] != str2[i]:
                mismatch += 1
                if mismatch >= 3:
                    break
        return len(str1) == len(str2) and mismatch < 3
    
    def calculate_first_rank(self, character, block, offset):
        return self.Ranks[character][block] - self.bwt[block][offset:].count(character) + 1

    def calculate_last_rank(self, character, block, offset):
        return self.Ranks[character][block] - self.bwt[block].count(character) + self.bwt[block][:offset + 1].count(character)

    def update_search_band(self, character, first_rank, last_rank):
        if character == 'A':
            return [first_rank - 1, last_rank - 1]
        elif character == 'C':
            return [self.Ranks['A'][-1] + first_rank - 1, self.Ranks['A'][-1] + last_rank - 1]
        elif character == 'G':
            return [self.Ranks['A'][-1] + self.Ranks['C'][-1] + first_rank - 1,
                    self.Ranks['A'][-1] + self.Ranks['C'][-1] + last_rank - 1]
        elif character == 'T':
            return [self.Ranks['A'][-1] + self.Ranks['C'][-1] + self.Ranks['G'][-1] + first_rank - 1,
                    self.Ranks['A'][-1] + self.Ranks['C'][-1] + self.Ranks['G'][-1] + last_rank - 1]
    
    def search(self, read):
        # Initial Band
        Band = [0, self.Ranks['A'][-1] + self.Ranks['C'][-1] + self.Ranks['G'][-1] + self.Ranks['T'][-1] + 1]

        # Search for the pattern in reverse order
        for i in reversed(range(len(read))):
            # Get the character to search for
            current_character = read[i]

            # Calculate the nearest blocks within the search band
            start_block, start_offset = divmod(Band[0], 100)
            end_block, end_offset = divmod(Band[1], 100)

            # Initialize variables to store the ranks of the first and last occurrences of the character
            first_rank, last_rank = -1, -1

            # Search for the first occurrence of the character within the search band
            first_rank = self.calculate_first_rank(current_character, start_block, start_offset)

            # Search for the last occurrence of the character within the search band
            last_rank = self.calculate_last_rank(current_character, end_block, end_offset)

            # If the last rank is smaller than the first rank, it means no match was found
            if last_rank < first_rank:
                return Band, i + 1

            # Update the search band based on the character
            Band = self.update_search_band(current_character, first_rank, last_rank)
        # Return the final search band and 0 to indicate no mismatch within the pattern
        return Band, 0

    def extractfromref(self, index, length):
        block = index // 100
        result = self.ref[index // 100][index % 100:-1]
        while len(result) < length:
            block += 1
            result += self.ref[block][:-1]
        return result[:length]
    
    def findRedGreenMatches(self, ReadComplemented):
        RedMatches, GreenMatches = np.zeros(6), np.zeros(6)
        for r in ReadComplemented:
                    band, off = self.search(r)
                    for i in range(band[0], band[1] + 1):
                        idx = int(self.indices[i]) - off
                        ref = self.extractfromref(idx, len(r))
                        if self.Match(ref, r):
                            for j in range(6):
                                if (idx >= self.exon_ranges[0][j] and idx <= self.exon_ranges[1][j]):
                                    RedMatches[j] = 1
                                if (idx >= self.exon_ranges[2][j] and idx <= self.exon_ranges[3][j]):
                                    GreenMatches[j] = 1
        return RedMatches, GreenMatches
    
    def process_reads(self, reads_file):
        with open(reads_file, 'r') as file:
            for read in file:
                read = read[:-1].replace('N', 'A')
                reversed_read = read[::-1]
                complemented_read = reversed_read.translate(str.maketrans('ATCG', 'TAGC'))
                Red, Green = self.findRedGreenMatches((read, complemented_read))
                self.update_exons(Red, Green)

    def update_exons(self, RedMatches, GreenMatches):
        for i in range(6):
            if GreenMatches[i] == 1 and RedMatches[i] == 1:
                self.RExons[i] += 0.5
                self.GExons[i] += 0.5
            elif RedMatches[i] == 1:
                self.RExons[i] += 1
            elif GreenMatches[i] == 1:
                self.GExons[i] += 1

    def set_gene_exon_ranges(self, exon_ranges):
        self.exon_ranges = exon_ranges

    def print_results(self):
        print("Red Exons:", self.RExons)
        print("Green Exons:", self.GExons)
    
    def possible_config(self, configs):
        division_result = np.array(self.RExons[1:-1]) / np.array(self.GExons[1:-1])
        cosine_similarities = cosine_similarity([division_result], configs)
        scores = cosine_similarities[0]
        softmax_scores = softmax(scores)
        return softmax_scores


if __name__ == "__main__":
    start_time = time.time()
    analysis = GeneticAnalysis(data_dir='../data')
    analysis.set_gene_exon_ranges([[149249757, 149256127, 149258412, 149260048, 149261768, 149264290],
                                   [149249868, 149256423, 149258580, 149260213, 149262007, 149264400],
                                   [149288166, 149293258, 149295542, 149297178, 149298898, 149301420],
                                   [149288277, 149293554, 149295710, 149297343, 149299137, 149301530]])
    analysis.process_reads(os.path.join(analysis.data_dir, 'reads'))
    analysis.print_results()
    configs = [[0.5, 0.5, 0.5, 0.5],
                    [1, 1, 0, 0],
                    [0.33, 0.33, 1, 1],
                    [0.33, 0.33, 0.33, 1]]
    print('Configs:\n', configs)
    probabilities = analysis.possible_config(np.array(configs))
    print("Scores of configs: ", probabilities)
    max_prob_index = np.argmax(probabilities)
    best_config = configs[max_prob_index]
    max_probability = probabilities[max_prob_index]
    print('Best possible config:', best_config, 'with probability', max_probability)
    end_time = time.time()
    print(f"Time taken to run the code: {end_time-start_time:.2f} seconds")