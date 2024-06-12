from thefuzz import fuzz
import multiprocessing 

def get_range(max_number,number_of_range):
    step=int(max_number/number_of_range)
    start=0
    result=[]
    for i in range(number_of_range):
        if i==number_of_range-1:
            result.append((start,max_number))
            break
        result.append((start,min(start+step,max_number)))
        start=min(start+step,max_number)
    return result

    
projects=['dagger','dubbo','ExoPlayer','flink','guava','kafka','spring-boot','spring-framework','spring-security']
w_2=open(f'../data_RQ3/train/train.token.truncated.code', 'r').readlines()
w_1=open(f'../dataset_v2/original/all_truncated.code', 'r').readlines()
w_3=open(f'../dataset_v2/original/all_truncated.sbt', 'r').readlines()
w_4=open(f'../data_RQ3/train/train.token.truncated.ast', 'r').readlines()
delete_one=set()
def get_delete(q,start,end,w_1,w_2):
    result=[]
    for i in range(start,end):
        for idx,line in enumerate(w_1):
            s=fuzz.ratio(w_2[i], line)
            if s>=90:
                result.append((i,idx))
    q.put(result)

#[(0, 57721), (57721, 115442), (115442, 173163), (173163, 230884), (230884, 288605), (288605, 346327)]
list_process=[]
list_range=get_range(346327,24)
q=multiprocessing.Queue()
for i in range(24):
    start,end=list_range[i]
    p=multiprocessing.Process(target=get_delete,args=(q,start,end,w_1,w_2,))
    print("Run process ",i)
    p.start()
    list_process.append(p)
for i in range(24):
    list_process[i].join() 
    res=q.get()
    for ele in res:
        delete_one.add(ele)
    print("Finish process ",i)

w_5=open(f'../data_RQ3/train/delete.txt', 'w')
for element in delete_one:
    first,second=element
    w_5.write(str(first)+' '+str(second)+'\n')      
print("Total duplicate ",len(delete_one))