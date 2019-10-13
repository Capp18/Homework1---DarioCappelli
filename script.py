# ===== PROBLEM1 =====
# Exercise 1 - Introduction - Say "Hello, World!" With Python
print("Hello, World!")
# Exercise 2 - Introduction - Python If-Else
import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
if n%2!=0:
    print("Weird")
if n%2==0:
    if (2<=n<=5)or(n>20):
        print("Not Weird")
    if(6<=n<=20):
        print("Weird")
# Exercise 3 - Introduction - Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
print(a+b)
print(a-b)
print(a*b)
# Exercise 4 - Introduction - Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
print(a//b)
print(a/b)
# Exercise 5 - Introduction - Loops
if __name__ == '__main__':
    n = int(input())

for i in range(n):
    print(i**2)
# Exercise 6 - Introduction - Write a function
def is_leap(year):
    if year%4==0:
        if year%100==0:
            if year%400==0:
                leap= True
            else:
                leap= False
        else:
            leap= True
    else:
        leap= False
    
    return leap
# Exercise 7 - Introduction - Print Function
if __name__ == '__main__':
    n = int(input())

for i in range(1,n+1):
    print(i,end='')
# Exercise 8 - Basic data types - List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())

l=[]
for i in range(x+1):
    for j in range(y+1):
        for k in range(z+1):
            if (i+j+k)!=n:
                l.append([i,j,k])
print(l)
# Exercise 9 - Basic data types - Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
l=[]
for el in arr:
        l.append(el) 
l1=[]
for el in l:
    if el<max(l):
        l1.append(el)
print(max(l1))
# Exercise 10 - Basic data types - Nested Lists
if __name__ == '__main__':
    l=[]
    l1=[]
    for _ in range(int(input())):
        name = input()
        score = float(input())
        l.append([name,score])
        l1.append(score)
    l.sort()            #ordino la lista principale
    l2=[]
    for i in range(len(l)):   
        if l[i][1]>min(l1):
            l2.append(l[i][1])     #creo una lista senza il minimo score
    for j in range(len(l)):
        if l[j][1]==min(l2):
            print(l[j][0])          #stampo i nomi che coincidono con lo score minimo della nuova lista

# Exercise 11 - Basic data types - Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()

l=student_marks.get(query_name)
print(format((l[0]+l[1]+l[2])/3,".2f"))
# Exercise 12 - Basic data types - Lists
if __name__ == '__main__':
    N = int(input())
    l=[]
    for i in range(1,N+1):
        order= input().split()
        if order[0]=="insert":
            l.insert(int(order[1]),int(order[2]))
        elif order[0]=="print":
            print(l)
        elif order[0]=="remove":
            l.remove(int((order[1])))
        elif order[0]=="append":
            l.append(int(order[1]))
        elif order[0]=="sort":
            l.sort()
        elif order[0]=="pop":
            l.pop(-1)
        elif order[0]=="reverse":
            l.reverse()

# Exercise 13 - Basic data types - Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
t=tuple(integer_list)
print(hash(t))
# Exercise 14 - Strings - sWAP cASE
def swap_case(s):
    s1=""
    for i in range(len(s)):
        if s[i].islower()==True:
            s1=s1+s[i].upper()
        else:
            s1=s1+s[i].lower()
    return s1
# Exercise 15 - Strings - String Split and Join
def split_and_join(line):
    line=line.split(" ")
    line="-".join(line)
    return line
# Exercise 16 - Strings - What's Your Name?
def print_full_name(a, b):
    print("Hello ",first_name," ",last_name,"! You just delved into python.",sep="")
# Exercise 17 - Strings - Mutations
def mutate_string(string, position, character):
    l=list(string)
    l[position]=character
    return ''.join(l)
# Exercise 18 - Strings - Find a string
def count_substring(string, sub_string):
    n=0
    for i in range(len(string)):
        if string[i:].startswith(sub_string)==True:
            n=n+1
    return n
# Exercise 19 - Strings - String Validators
if __name__ == '__main__':
    s = input()

x=False
for chr in s:
    if chr.isalnum()==True:
        x=True
        break
print(x)

x1=False
for chr in s:
    if chr.isalpha()==True:
        x1=True
        break
print(x1)

x2=False
for chr in s:
    if chr.isdigit()==True:
        x2=True
        break
print(x2)

x3=False
for chr in s:
    if chr.islower()==True:
        x3=True
        break
print(x3)

x4=False
for chr in s:
    if chr.isupper()==True:
        x4=True
        break
print(x4)
# Exercise 20 - Strings - Text Alignment
thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))
# Exercise 21 - Strings - Text Wrap
import textwrap

def wrap(string, max_width):
    return textwrap.fill(string,max_width)

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)
# Exercise 22 - Strings - Designer Door Mat
n,m=map(int,input().split())
x=".|."
for i in range(0,n-1,2):
    print (((x*i)+x).center(m,'-'))
print(("WELCOME").center(m,"-"))
i=n-2
while(i>=0):
    print((x*i).center(m,"-"))
    i=i-2
# Exercise 23 - Strings - String Formatting
def print_formatted(number):
    w=len(bin(n).replace("0b",""))
    for i in range(1,n+1):
        d=str(i).rjust(w," ")
        o=oct(i).replace("0o","").rjust(w," ")
        e=(hex(i).replace("0x","")).upper().rjust(w," ")
        b=bin(i).replace("0b","").rjust(w," ")
        print(d,o,e,b)
    return 
# Exercise 24 - Strings - Alphabet Rangoli
def print_rangoli(size):
    w=(((size*2)-1)*2)-1
    import string
    alp=list(string.ascii_lowercase[:])
    for i in range(size):
        s=alp[size-i-1]    #lettera centrale
        j=i
        while(j>0):
            s=alp[size-j]+"-"+s+"-"+alp[size-j]
            j=j-1
        s=s.center(w,"-")

        print(s)  #FIN PRIMA PARTE
    i=size-2
    while i>=0:
        s=alp[size-i-1]
        j=i
        while(j>0):
            s=alp[size-j]+"-"+s+"-"+alp[size-j]
            j=j-1
        s=s.center(w,"-")
        print(s)
        i=i-1
    return
# Exercise 25 - Strings - Capitalize!
def solve(s):
    s1=s[0].capitalize()
    for i in range(1,len(s)):
        if s[i-1]==" ":
            s1=s1+s[i].capitalize()
        else:
            s1=s1+s[i]
    return s1
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()
# Exercise 26 - Strings - The Minion Game
# Exercise 27 - Strings - Merge the Tools!
def merge_the_tools(string, k):
    i=0
    while i<=len(string):
        t=string[i:(i+k)]
        u=""
        for chr in t:
            if chr not in u:
                u+=chr
            else:
                pass
        print(u)
        i=i+k
    return
        
if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)
# Exercise 28 - Sets - Introduction to Sets
def average(array):
    l=[]
    for i in range(n):
        if (arr[i] in  l)==False:
            l.append(arr[i])
    return (sum(l)/len(l))
# Exercise 29 - Sets - No Idea!
n,m=map(int,input().split())
arr=list(map(int,input().split()))
A=set(map(int,input().split()))
B=set(map(int,input().split()))

happiness=0
for i in range(n):
    if (arr[i] in A)==True:
        happiness=happiness+1
    if(arr[i] in B)==True:
        happiness=happiness-1
print(happiness)
# Exercise 30 - Sets - Symmetric Difference
m=int(input())
l1=set(map(int,input().split()))
n=int(input())
l2=set(map(int,input().split()))
ris=list(l1^l2)
ris.sort()
for i in range(len(ris)):
    print(ris[i])
# Exercise 31 - Sets - Set .add()
n=int(input())
s=set()
for i in range(n):
    s.add(input())
print(len(s))
# Exercise 32 - Sets - Set .discard(), .remove() & .pop()
n=int(input())
s=set(map(int, input().split()))
N=int(input())
for i in range(N):
    l=input().split()
    if l[0]=="pop":
        s.pop()
    elif l[0]=="remove":
        s.remove(int(l[1]))
    elif l[0]=="discard":
        s.discard(int(l[1]))
print(sum(s))
# Exercise 33 - Sets - Set .union() Operation
n=int(input())
en=set(map(int,input().split()))
b=int(input())
fr=set(map(int,input().split()))
print(len(en|fr))
# Exercise 34 - Sets - Set .intersection() Operation
n=int(input())
en=set(map(int,input().split()))
b=int(input())
fr=set(map(int,input().split()))
print(len(en&fr))
# Exercise 35 - Sets - Set .difference() Operation
n=int(input())
en=set(map(int,input().split()))
b=int(input())
fr=set(map(int,input().split()))
print(len(en-fr))
# Exercise 36 - Sets - Set .symmetric_difference() Operation
n=int(input())
en=set(map(int,input().split()))
b=int(input())
fr=set(map(int,input().split()))
print(len(en^fr))
# Exercise 37 - Sets - Set Mutations
n=int(input())
A=set(map(int,input().split()))
N=int(input())
for i in range(N):
    l=input().split()
    B=set(map(int,input().split()))
    if l[0]=="intersection_update":
        A.intersection_update(B)
    elif l[0]=="update":
        A.update(B)
    elif l[0]=="symmetric_difference_update":
        A.symmetric_difference_update(B)
    elif l[0]=="difference_update":
        A.difference_update(B)
print(sum(A))
# Exercise 38 - Sets - The Captain's Room
from collections import Counter
K=int(input())
l=list(map(int,input().split()))
d=Counter(l)
for el in d:
    if d[el]==1:
        print(el)
# Exercise 39 - Sets - Check Subset
for i in range(int(input())):
    na=int(input())
    A=list(map(int,input().split()))
    nb=int(input())
    B=list(map(int,input().split()))
    n=0
    if na<nb:
        for el in A:
            if (el in B)==True:
                n+=1
        if n==len(A):
            print(True)
        else:
            print(False)
    else:
        print(False)
# Exercise 40 - Sets - Check Strict Superset
A=list(map(int,input().split()))
n=int(input())
ok=0
for i in range(n):
    s=list(map(int,input().split()))
    t=0
    for el in s:
        if (el in A)==True:
            t+=1
    if t==len(s):
        ok+=1
if ok==n:
    print(True)
else:
    print(False)

# Exercise 41 - Collections - collections.Counter()
X=int(input())
shoes=list(map(int,input().split()))
N=int(input())
cash=0
for i in range(N):
    cos=list(map(int,input().split()))
    if (cos[0] in shoes)==True:
        cash=cash+cos[1]
        shoes.remove(cos[0])
print(cash)
# Exercise 42 - Collections - DefaultDict Tutorial
from collections import defaultdict
inp=list(map(int,input().split()))
A=defaultdict(list)
l=[]
for i in range(1,inp[0]+1):
    a=input()
    A[a].append(i)
for i in range(inp[1]):
    l.append(input())
for i in l:
    if i in A:
        print(" ".join(map(str,A[i])))
    else:
        print(-1)
# Exercise 43 - Collections - Collections.namedtuple()
from collections import namedtuple
n=int(input())
stud=namedtuple('stud',input())
l=[]
for i in range(n):
    a=list(map(str,input().split()))
    x=stud(a[0],a[1],a[2],a[3])
    l.append(int(x.MARKS))
print(sum(l)/len(l))
# Exercise 44 - Collections - Collections.OrderedDict()
from collections import OrderedDict
d=OrderedDict()
for i in range(int(input())):
    a=list(map(str,input().split()))
    item=" ".join(a[0:-1])
    if item not in d:
        d[item]=int(a[-1])
    else:
        d[item]+=int(a[-1])
for el in d:
    print(el,d[el])
# Exercise 45 - Collections - Word Order
from collections import OrderedDict
d=OrderedDict()
for i in range(int(input())):
    a=input()
    if a not in d:
        d[a]=1
    else:
        d[a]+=1
print(len(d))
for el in d:
    print(d[el],end=" ")
# Exercise 46 - Collections - Collections.deque()
from collections import deque
d=deque()
for i in range(int(input())):
    c=list(map(str,input().split()))
    if(c[0]=="append"):
        d.append(c[1])
    elif(c[0]=="appendleft"):
        d.appendleft(c[1])
    elif(c[0]=="pop"):
        d.pop()
    elif(c[0]=="popleft"):
        d.popleft()
for el in d:
    print(el,end=" ")
# Exercise 47 - Collections - Company Logo
import math
import os
import random
import re
import sys
from collections import Counter
if __name__ == '__main__':
    s = input()
l=[]
for chr in s:
    l.append(chr)
l.sort()
d=Counter(l)
for el in d.most_common(3):
    print(el[0],el[1])

# Exercise 48 - Collections - Piling Up!
from collections import deque
for _ in (range(int(input()))):
    input()
    s = deque(map(int, input().strip().split()))
    result = "Yes"
    if max(s) not in (s[0],s[-1]):
        result = "No"
    print(result)
    #I got some help from discussion because i didn't understand the task
# Exercise 49 - Date time - Calendar Module
import calendar
date=list(map(int,input().split()))
a=calendar.weekday(date[2],date[0],date[1])
week=['MONDAY','TUESDAY','WEDNESDAY','THURSDAY','FRIDAY','SATURDAY','SUNDAY']
print(week[a])
# Exercise 50 - Date time - Time Delta
import math
import os
import random
import re
import sys
from datetime import datetime

# I studied the solution from discussion
def time_delta(t1, t2):
    s1=datetime.strptime(t1,'%a %d %b %Y %H:%M:%S %z')
    s2=datetime.strptime(t2,'%a %d %b %Y %H:%M:%S %z')
    return str(int(abs((s1-s2).total_seconds())))
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()

# Exercise 51 - Exceptions -
T=int(input())
for i in range(T):
    try:
        n=list(map(int,input().split()))
        print(int(n[0])//int(n[1]))
    except Exception as er:
        print("Error Code:",er)
# Exercise 52 - Built-ins - Zipped!
N,X=map(int,input().split())
b=[]
for i in range(X):
    a=list(map(float,input().split()))
    b.append(a)
for i in zip(*b):
    print(sum(i)/len(i))
# Exercise 53 - Built-ins - Athlete Sort
import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    
arr.sort(key=lambda x: x[k])
for line in arr:
    print(*line, sep=' ')

# Exercise 54 - Built-ins - Ginorts
s=input()
s1=""
s2=""
s3=""
s4=""
for chr in s:
    if chr.islower()==True:
        s1+=chr
    elif chr.isupper()==True:
        s2+=chr
    elif chr.isdigit()==True:
        if(int(chr)%2!=0)==True:
            s3+=chr
        else:
            s4+=chr

print("".join(sorted(s1))+"".join(sorted(s2))+"".join(sorted(s3))+"".join(sorted(s4)))
# Exercise 55 - Map and lambda function
cube = lambda x: x**3
def fibonacci(n):
    m=[]
    if n>=1:
        m.append(0)
    if n>=2:
        m.append(1)
        for i in range(2,n):
            m.append(m[i-2]+m[i-1])
    return m
#For regex I often used discussion for help, because I have never seen the command before 
# Exercise 56 - Regex - Detect Floating Point Number
import re
N=int(input())
for i in range(N):
    x=input()
    print(bool(re.match(r"^[-+]?[0-9]*\.[0-9]+$",x)))
# Exercise 57 - Regex - Re.split()
regex_pattern = r"[,.]"
# Exercise 58 - Regex - Group(), Groups() & Groupdict()
import re
s=input()
l=re.search(r'([a-zA-Z0-9])\1+',s)
if l:
    print(l.group(1))
else:
    print(-1)
# Exercise 59 - Regex - Re.findall() & Re.finditer()
import re
alp= '[qwrtypsdfghjklzxcvbnm]'
s=input()
b = re.findall('(?<=' + alp +')([aeiou]{2,})' + alp, s, re.I)
print('\n'.join(b or ['-1']))
# Exercise 60 - Regex - Re.start() & Re.end()
import re
s=input()
k=input()
a = list(re.finditer(r'(?={})'.format(k),s))
if a:
    for el in a:
        print(''.join(str((el.start(),el.start()+(len(k)-1)))))
else:
    print('(-1, -1)')
# Exercise 61 - Regex - Regex Substitution
# Exercise 62 - Regex - Validating Roman Numerals
# Exercise 63 - Regex - Validating phone numbers
# Exercise 64 - Regex - Validating and Parsing Email Addresses
# Exercise 65 - Regex - Hex Color Code
# Exercise 66 - Regex - HTML Parser - Part 1
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def _attr_printer(self, attrs):
        if attrs:
            for attr in attrs:
                print("-> " + attr[0] + " > " + str(attr[1]) )
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        self._attr_printer(attrs)
    def handle_endtag(self, tag):
        print("End   :", tag)
    def handle_startendtag(self, tag, attrs):
        print("Empty :", tag)
        self._attr_printer(attrs) 
n = int(input())
h=""
for i in range(n):
    s=input()
    h+=s
p=MyHTMLParser()
p.feed(h)
# Exercise 67 - Regex - HTML Parser - Part 2
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if len(data.split('\n')) == 1:
            print('>>> Single-line Comment')
            print(data)
        else:
            print('>>> Multi-line Comment')
            print(data)
    def handle_data(self, data):
        if data != '\n':
            print('>>> Data')
            print(data)
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()
# Exercise 68 - Regex - Detect HTML Tags, Attributes and Attribute Values
# Exercise 69 - Regex - Validating UID
# Exercise 70 - Regex - Validating Credit Card Numbers
# Exercise 71 - Regex - Validating Postal Codes
# Exercise 72 - Regex - Matrix Script
# Exercise 73 - Xml - XML 1 - Find the Score
import sys
import xml.etree.ElementTree as etree
def get_attr_number(node):
    return etree.tostring(node).count(b'=')
if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))
# Exercise 74 - Xml - XML 2 - Find the Maximum Depth
# Exercise 75 - Closures and decorators - Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
            f(["+91"+" "+c[-10:-5]+" "+c[-5:] for c in l])
    return fun
# Exercise 76 - Closures and decorators - Decorators 2 - Name Directory
def person_lister(f):
    def inner(people):
        for person in sorted(people,key=lambda x:int(x[2])):
            yield f(person)
    return inner
# Exercise 77 - Numpy - Arrays
def arrays(arr):

    l=list(arr)
    l.reverse()
    a=numpy.array(l,float)
    return a
# Exercise 78 - Numpy - Shape and Reshape
import numpy
arr= numpy.array(list(map(int,input().split())))
arr.shape=(3,3)
print(arr)
# Exercise 79 - Numpy - Transpose and Flatten
import numpy
inp=list(map(int,input().split()))
N=inp[0]
M=inp[1]
lis=[]
for i in range(N):
    lis.append(list(map(int,input().split())))
arr=numpy.array(lis)
print(numpy.transpose(arr))
print(arr.flatten())
# Exercise 80 - Numpy - Concatenate
import numpy
inp=list(map(int,input().split()))
N=inp[0]
M=inp[1]
P=inp[2]
l1=[]
l2=[]
for i in range(N):
    l1.append(list(map(int,input().split())))
for i in range(M):
    l2.append(list(map(int,input().split())))
arr1=numpy.array(l1)
arr2=numpy.array(l2)
print(numpy.concatenate((arr1,arr2),axis=0))

# Exercise 81 - Numpy - Zeros and Ones
import numpy
inp=tuple(map(int,input().split()))
print(numpy.zeros(inp,dtype=numpy.int))
print(numpy.ones(inp,dtype=numpy.int))
# Exercise 82 - Numpy - Eye and Identity
import numpy
inp=(list(map(int,input().split())))
numpy.set_printoptions(sign=" ")   #space between matrix's element
print(numpy.eye(inp[0], inp[1]))
# Exercise 83 - Numpy - Array Mathematics
import numpy
n,m=map(int,input().split())
l1=[]
for i in range(n):
    l1.append(list(map(int,input().split())))
l2=[]
for i in range(n):
    l2.append(list(map(int,input().split())))
arr1=numpy.array(l1)
arr2=numpy.array(l2)
print(arr1+arr2)
print(arr1-arr2)
print(arr1*arr2)
print(arr1//arr2)
print(arr1%arr2)
print(arr1**arr2)

# Exercise 84 - Numpy - Floor, Ceil and Rint
import numpy
arr=numpy.array(list(map(float,input().split())))
numpy.set_printoptions(sign=" ")
print(numpy.floor(arr))
print(numpy.ceil(arr))
print(numpy.rint(arr))
# Exercise 85 - Numpy - Sum and Prod
import numpy
n,m=map(int,input().split())
l=[]
for i in range(n):
    l.append(list(map(int,input().split())))
arr=numpy.array(l)
s=numpy.sum(arr,axis=0)
print(numpy.prod(s))

# Exercise 86 - Numpy - Min and Max
import numpy
n,m=map(int,input().split())
l=[]
for i in range(n):
    l.append(list(map(int,input().split())))
arr=numpy.array(l)
print(numpy.max(numpy.min(arr,axis=1)))
# Exercise 87 - Numpy - Mean, Var, and Std
import numpy
n,m=map(int,input().split())
l=[]
for i in range(n):
    l.append(list(map(int,input().split())))
arr=numpy.array(l)
numpy.set_printoptions(sign=" ")
numpy.set_printoptions(legacy="1.13")
print(numpy.mean(arr,axis=1))
print(numpy.var(arr,axis=0))
print(numpy.std(arr))

# Exercise 88 - Numpy - Dot and Cross
import numpy as n
m=int(input())
l1=[]
l2=[]
for i in range(m):
    l1.append(list(map(int,input().split())))
for i in range(m):
    l2.append(list(map(int,input().split())))
arr1=n.array(l1)
arr2=n.array(l2)
print(n.dot(arr1,arr2))
# Exercise 89 - Numpy - Inner and Outer
import numpy as n
a=n.array(list(map(int,input().split())))
b=n.array(list(map(int,input().split())))
print(n.inner(a,b))
print(n.outer(a,b))
# Exercise 90 - Numpy - Polynomials
import numpy as n
P=n.array(list(map(float,input().split())))
x=int(input())
print(n.polyval(P,x))
# Exercise 91 - Numpy - Linear Algebra
import numpy as n
a=int(input())
l=[]
for i in range(a):
    l.append(list(map(float,input().split())))
arr=n.array(l)
n.set_printoptions(legacy="1.13")
print(n.linalg.det(arr))

# ===== PROBLEM2 =====

# Exercise 92 - Challenges - Birthday Cake Candles
import math
import os
import random
import re
import sys

# Complete the birthdayCakeCandles function below.
def birthdayCakeCandles(ar):
    return ar.count(max(ar))
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    ar_count = int(input())

    ar = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(ar)

    fptr.write(str(result) + '\n')

    fptr.close()
# Exercise 93 - Challenges - Kangaroo
import math
import os
import random
import re
import sys

# Complete the kangaroo function below.
def kangaroo(x1, v1, x2, v2):
    for  i in range(1000000):
        if (x1+v1*i)==(x2+v2*i):
            s="YES"
            break
        else:
            s="NO"
    return s
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()
# Exercise 94 - Challenges - Viral Advertising
import math
import os
import random
import re
import sys


def viralAdvertising(n):
    likes=0
    people=5
    for i in range(n):
        likes=likes+(people//2)
        people=(people//2)*3
    return likes

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

# Exercise 95 - Challenges - Recursive Digit Sum
import math
import os
import random
import re
import sys

def superDigit(p):
    j=0
    if len(p)==1:
        return p
    for i in range(len(p)):
        j+=int(p[i])
    p=str(j)
    return superDigit(p)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(str(nk[1][0]))
    p=n*k
    result = superDigit(p)

    fptr.write(str(result) + '\n')

    fptr.close()
# Exercise 96 - Challenges - Insertion Sort - Part 1
# Exercise 97 - Challenges - Insertion Sort - Part 2
