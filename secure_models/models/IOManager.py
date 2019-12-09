import csv, time

class IO():
    def __init__(self):
        self.lines = []
        self.start_time = time.time()
    
    def finish(self):
        with open("addresses.csv", 'a') as writeFile:
            writer = csv.writer(writeFile)
            writer.writerows(self.lines)
        writeFile.close()
    
    def write_address(self, addr, dtype, read = True):
        t = time.time() - self.start_time
        if read:
            self.lines.append(["Read", t, addr, dtype])
        else:
            self.lines.append(["Write", t, addr, dtype])
    
    def write(self, addr, dtype, read = True):
        t = time.time() - self.start_time
        if read:
            self.lines.append(["R", t, addr])
        else:
            self.lines.append(["W", t, addr])
    
    def do_nothing(self):
        pass