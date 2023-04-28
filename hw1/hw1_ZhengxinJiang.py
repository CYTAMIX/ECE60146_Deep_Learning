# ECE60146 HW1
# Zhengxin Jiang
# jiang839

class Sequence(object):
    def __init__(self, array):
        self.array = array

    def __iter__(self):
        self.idx = -1
        return self
    
    def __next__(self):
        self.idx += 1
        
        if self.idx<len(self.array):   
            return self.array[self.idx]
        else:
            raise StopIteration
            
    def __len__(self):
         return len(self.array)
        
    def __gt__(self, other):
        # check if two objects have the same length
        if len(self.array) != len(other.array):
            raise ValueError('Two arrays are not equal in length !')
        
        count = 0
        for i in range(len(self.array)):
            if self.array[i] > other.array[i]:
                count += 1
                
        return count
        

# subclass
class Fibonacci(Sequence):
    
    def __init__(self, first_value, second_value):
        Sequence.__init__(self, [first_value, second_value])
        
        
    def __call__(self, length): 
        # init
        self.array = self.array[:2]
        
        if length > 2:
            for i in range(2, length):
                self.array.append(self.array[i-1]+self.array[i-2])
                
        print(self.array)
            
# subclass
class Prime(Sequence):
    
    def __init__(self):
        Sequence.__init__(self, [])       
    
    def __call__(self, length): 
        # init
        self.array = []
        
        if length == 1:
            self.array.append(2)
        
        if length > 1:
            self.array.append(2)
            
            num = 3
            while len(self.array)<length:
                is_prime = True

                # test if a number is prime
                for i in range(2, num):
                    if num%i == 0:
                        is_prime = False
                        break
                    
                if is_prime:
                    self.array.append(num)
                    
                num += 1

        print(self.array)
        
        
# Main
# Task 1-4
FS = Fibonacci(1,2)
FS(length = 5)
print(len(FS))
print([n for n in FS])

# Task 5
PS = Prime()
PS(length = 8)
print(len(PS))
print([n for n in PS])

# Task 6
FS = Fibonacci(1,2)
FS(length = 8)
PS = Prime()
PS(length = 8)
print(FS>PS)
PS(length = 5)
print(FS>PS)
