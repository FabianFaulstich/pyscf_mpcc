import numpy as np 

if __name__ == "__main__":

    file = open('Geometries.txt', 'r')
    Lines = file.readlines()
     
    count = 0
    IRCs = []
    first_file = True
    # Strips the newline character
    for line in Lines:
        if line[:3] == "IRC":
            IRC = float(line[6:-1])
            IRCs.append(IRC)
            name = f'Fenton_{IRC}'
            if first_file:
                first_file = False
                geom_file = open(name, 'w')
            else:
                geom_file.close()
                geom_file = open(name, 'w')

        else:
            geom_file.write(line)
            #line_str = line.split()
            #element = line_str[0]
            #geometry = [float(line_str[1]), float(line_str[2]), float(line_str[3])]
   
    IRCs = np.array(IRCs)
    np.save('IRCs.npy', IRCs)
