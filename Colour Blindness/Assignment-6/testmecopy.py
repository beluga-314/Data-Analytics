import numpy as np
import time
import os

class GeneticAnalysis:

    def __init__(self, data_dir='../data'):
        self.data_dir = data_dir
        self.bwt, self.Ranks = self.load_data(os.path.join(data_dir, 'chrX_last_col.txt'), ranks=True)
        self.indices = self.load_data(os.path.join(data_dir, 'chrX_map.txt'))
        self.ref = self.load_data(os.path.join(data_dir, 'chrX.fa'))
        # Initialize counts for match in each exon of Red and Green genes
        self.RExons = np.zeros(6)
        self.GExons = np.zeros(6)
        self.Exons = np.zeros(6)

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

    def extractfromref(self, index, length):
        block = index // 100
        result = self.ref[index // 100][index % 100:index % 100+length]
        while len(result) < length:
            block += 1
            result += self.ref[block][:length - len(result)]
        return result
    
    def search(self,read):
    #Initial Band
        Band = [0,self.Ranks['A'][-1]+self.Ranks['C'][-1]+self.Ranks['G'][-1]+self.Ranks['T'][-1]+1]

        #Searching in reverse for each suffix
        for i in reversed(range(len(read))):        
            #Character C to be searches
            C = read[i]

            #Generating nearest Blocks
            start,sidx = divmod(Band[0], 100)
            end,eidx = divmod(Band[1], 100)

            #Rank of first and last C
            rf1,rf2 = -1,-1
            
            #Searching for first C in Band
            rf1 = self.calculate_first_rank(C, start, sidx)

            #Searching for last C in Band
            rf2 = self.calculate_last_rank(C, end, eidx)

            if rf2<rf1:
                return Band,i+1

            Band = self.update_search_band(C, rf1, rf2)
        return Band,0

    def reverseComplement(self, read:str):
        read = read[::-1]
        result = read.replace('A','t').replace('T','a').replace('C','g').replace('G','c')
        return str.upper(result) 
    def process_reads(self, reads_file):
        with open(reads_file,'r') as file:
            for i, read in enumerate(file):
                #Replacing N with A in reads
                if i % 10000 == 0:
                    print(i)
                    print(self.RExons)
                    print(self.GExons)
                read = read[:-1].replace('N','A')

                #Reverse Complement of string
                readrevcomp = self.reverseComplement(read)

                #Indicators that read matches to R and G
                R,G = np.zeros(6),np.zeros(6)

                for read_ in [read,readrevcomp]:
                #Search Operation
                #Shift indicates the position in read where first mismatch occured from right to left
                    band,shift = self.search(read_)


                    for i in range(band[0],band[1]+1):
                        
                        #Adjusting index
                        id = int(self.indices[i])-shift
                        
                        #Extracting reference string from reference at index id
                        ref = self.extractfromref(id,len(read_))
                        
                        #Testing if mismatches between read and extracted string is < 2
                        if self.Match(ref,read_):
                        #Checking Red Gene and storing result for match in each Exon
                            if (id>=149249757 and id<=149249868):
                                R[0]=1
                            if (id>=149256127 and id<=149256423):
                                R[1]=1
                            if (id>=149258412 and id<=149258580):
                                R[2]=1
                            if (id>=149260048 and id<=149260213): 
                                R[3]=1
                            if (id>=149261768 and id<=149262007): 
                                R[4]=1
                            if (id>=149264290 and id<=149264400):
                                R[5]=1

                            #Checking Green Gene
                            if (id>=149288166 and id<=149288277): 
                                G[0]=1
                            if (id>=149293258 and id<=149293554): 
                                G[1]=1
                            if (id>=149295542 and id<=149295710):
                                G[2]=1
                            if (id>=149297178 and id<=149297343): 
                                G[3]=1
                            if (id>=149298898 and id<=149299137): 
                                G[4]=1
                            if (id>=149301420 and id<=149301530):
                                G[5]=1

                #Saving the matched result in global result
                for i in range(6):            
                    if R[i] == G[i] and R[i] == 1:
                        self.RExons[i] += 0.5
                        self.GExons[i] += 0.5
                    elif R[i] == 1:
                        self.RExons[i] += 1
                    elif G[i] == 1:
                        self.GExons[i] += 1

    def update_exons(self, RedMatches, GreenMatches):
        for i in range(6):
            if RedMatches[i] == GreenMatches[i] and RedMatches[i] == 1:
                self.Exons[i] += 0.5
            elif RedMatches[i] == 1 or GreenMatches[i] == 1:
                self.Exons[i] += 1
            if RedMatches[i] == 1:
                self.RExons[i] += 1
            if GreenMatches[i] == 1:
                self.GExons[i] += 1

    def set_gene_exon_ranges(self, exon_ranges):
        self.exon_ranges = exon_ranges

    def print_results(self):
        print("Exons:", self.Exons)
        print("RExons:", self.RExons)
        print("GExons:", self.GExons)

if __name__ == "__main__":
    start_time = time.time()
    analysis = GeneticAnalysis()
    analysis.set_gene_exon_ranges([[149249757, 149256127, 149258412, 149260048, 149261768, 149264290],
                                   [149249868, 149256423, 149258580, 149260213, 149262007, 149264400],
                                   [149288166, 149293258, 149295542, 149297178, 149298898, 149301420],
                                   [149288277, 149293554, 149295710, 149297343, 149299137, 149301530]])
    analysis.process_reads(os.path.join(analysis.data_dir, 'reads'))
    analysis.print_results()
    end_time = time.time()
    print(f"Time taken to run the code: {end_time-start_time:.2f} seconds")