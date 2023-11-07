print("1.Finding the maximum number in an array.\n")

a=[2,4,1,5,8]
print("array=",a)
max=a[0]
t=len(a)
for i in range(t):
    if a[i]>max:
        max=a[i]
print("Largest number:",max)
print("\n")


print("2. Find prime number in an array.\n")

arr=[1,28,4,8,6,48,53,10,19]
print("array=",arr)
p=len(arr)
print("Prime numbers are:")
for i in range(p):
    q=arr[i]
    flag=0
    for j in range(2,q):
        if arr[i]%j==0:
            flag=1
            break
    if flag==0:
        print(arr[i])
print("\n")


print("3. Implement a sorting algorithm.\n")

arr1=[8,3,6,91,4,5,1]
print("array=",arr1)
r=len(arr1)
tmp=0
for i in range(r-1):
    for j in range(0,r-i-1):
        if arr1[j]>arr1[j+1]:
            tmp=arr1[j]
            arr1[j]=arr1[j+1]
            arr1[j+1]=tmp
print("Sorted array=",arr1)


        
