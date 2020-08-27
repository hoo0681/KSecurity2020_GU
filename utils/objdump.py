import subprocess
import tqdm
import os

def test():
    # params = ['objdump', '-d', sample]
    # process = subprocess.Popen(params, stdout=subprocess.PIPE, universal_newlines=True)
    # #stdout, stderr = process.communicate()

    # # reader = csv.DictReader(stdout.splitlines(),
    # #                         delimiter='\t', skipinitialspace=True,
    # #                         fieldnames=['None']
    # #                         )
    # # for row in reader:
    # #     print(row)
    pass

def multiplepipe():
    dict_instructions = {}
    #sample = '/home/kisa/Documents/objdump/000aadad7b6e9316638e920f863855e7.vir'
    trainsetpath = '/home/kisa/Documents/trainset'

    for sample in tqdm.tqdm(os.listdir(trainsetpath)):    
        sample = os.path.join(trainsetpath, sample)
        p1_params = ["objdump -d {0} | awk -F '[\t]' '{{print $3}}' | awk -F ' ' '{{print $1}}'".format(sample)]
        p1 = subprocess.Popen(p1_params, stdout=subprocess.PIPE, shell=True)
        out, err = p1.communicate()

        list_instructions = out.decode('utf-8').rstrip('\n').splitlines()
        for idx, line in enumerate(list_instructions):
            if not line:
                continue
            
            else:
                if line in dict_instructions:
                    
                    #print(line)
                    value = dict_instructions.get(line)
                    value += 1
                    dict_instructions[line] = value
                else:

                    dict_instructions[line] = 0


    r = sorted(dict_instructions.items(), key=lambda t : t[1], reverse=True)
    
    for row in r:
        print(row)
    
    #print(sorted(dict_instructions))
    #read json instruction file and 

def main():
    multiplepipe()

if __name__ == '__main__':
    main()