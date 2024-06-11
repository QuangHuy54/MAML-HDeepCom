from thefuzz import fuzz

projects=['dagger','dubbo','ExoPlayer','flink','guava','kafka','spring-boot','spring-framework','spring-security']
w_2=open(f'../data_RQ3/train/train.token.code', 'r').readlines()
w_1=open(f'../dataset_v2/original/all_truncated_final.code', 'r').readlines()
w_3=open(f'../dataset_v2/original/all_truncated.sbt', 'r').readlines()
w_4=open(f'../data_RQ3/train/train.token.ast', 'r').readlines()
sum=0
delete_one=set()
for idx1,line1 in enumerate(w_2):
    for idx,line in enumerate(w_1):
            s = fuzz.ratio(line, line1)
            if s>90:
                sum+=1
                print(line)
                print(line1)
                print(w_3[idx])
                print(w_4[idx1])
                print(s)
                delete_one.add(idx1)
                break
w_5=open(f'../data_RQ3/train/delete.txt', 'w')
for element in delete_one:
    w_5.write(str(element)+'\n')      
print("Total duplicate ",sum)