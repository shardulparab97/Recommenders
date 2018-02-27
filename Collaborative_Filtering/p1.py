from math import sqrt
import math
import time
import pickle
recommendedList={}
ratings={}
rankings={}
#function to calculate pearson simlarity
def sim_pearson(prefs,p1,p2):

    # Finding the list of all the items which are simiar for p1 and p2
    si={}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item]=1
    # Find the number of elements
    n=len(si)
    # if they are no ratings in common, return 0
    if n==0:
        return 0
     # Add up all the ratings
    #my pearson score rating
    mean_p1=sum([prefs[p1][it] for it in prefs[p1]])/len(prefs[p1])
    mean_p2=sum([prefs[p2][it] for it in prefs[p2]])/len(prefs[p2])
    den_p1 = math.sqrt(sum([(math.pow((prefs[p1][it] - mean_p1 ),2) )for it in si]))
    den_p2 = math.sqrt(sum([(math.pow((prefs[p2][it] - mean_p2 ),2)) for it in si]))
    num = sum([((prefs[p1][it] - mean_p1)*(prefs[p2][it] - mean_p2)) for it in si])
    den =den_p1*den_p2
    if(den == 0): return 0
    r = num/den
    return r


# Using user ser collaborative filtering
def getRecommendations(prefs,person,similarity=sim_pearson):
    totals={}
    simSums={}
    for other in prefs:
    # compare all other users except himself
        if other==person: continue
    #finding similarity of the user with every other user
        sim=similarity(prefs,person,other)
    #not using less than or equal to zero similarity
        if sim<=0: continue
        for item in prefs[other]:
            # only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item]==0:
            # Similarity * Score
                #item here is the key
                totals.setdefault(item,0)
                totals[item]+=prefs[other][item]*sim
# Sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim
# Create the normalized list
    #rankings=[((total/simSums[item]),item) for item,total in totals.items( )]
    rankings[person]={}
    for item in totals:
    #rankings[person] = totals[item]/simSums[item]
        rankings[person][item]=totals[item]/simSums[item]


def loadMovieLens(path='./u.data'):
# Get movie titles
    movies={}
    #will use the movie title in the next loop
    for line in open('../data/u.item'):
        (id,title)=line.split('|')[0:2]
        movies[id]=title
# Load data
    prefs={}
    for line in open('../data/ua.base'):
        (user,movieid,rating,ts)=line.split('\t')
        prefs.setdefault(user,{})
        #setting key and its value as a dictionary
        #print prefs
        prefs[user][movies[movieid]]=float(rating)
    return prefs

def loadTestData(path='./u1.test'):
# Get movie titles
    movies={}
    #will use the movie title in the next loop
    for line in open('../data/u.item'):
        (id,title)=line.split('|')[0:2]
        movies[id]=title
# Load data
    prefs={}
    for line in open('../data/ua.test'):
        (user,movieid,rating,ts)=line.split('\t')
        prefs.setdefault(user,{})
        #setting key and its value as a dictionary
        #print prefs
        prefs[user][movies[movieid]]=float(rating)
    return prefs

#function for getting mean ratings of each user
def getMeanRatings(prefs):
    meanRatings = {}
    for user in prefs:
        count = 0
        sum = 0
        for item in prefs[user]:
            sum += prefs[user][item]
            count += 1
        meanRatings[user]=sum/count
    return meanRatings

#function to subtract mean ratings
def subMeanRatings(prefs,meanRatings):
    for user in prefs:
        for item in prefs[user]:
            prefs[user][item] -= meanRatings[user]

    return prefs

#function to again add mean ratings
def addMeanRatings(prefs,meanRatings):
    for user in prefs:
        for item in prefs[user]:
            prefs[user][item] += meanRatings[user]
    return prefs


meanRatings = getMeanRatings(loadMovieLens())
trainData=subMeanRatings(loadMovieLens(),meanRatings)


testData= subMeanRatings(loadTestData(),meanRatings)


start = time.clock()
for i in range(1,944):
    print (i)
    getRecommendations(trainData,str(i))
end = time.clock()

print (end - start)

addMeanRatings(rankings,meanRatings)

f=open('./p4.p','w')
pickle.dump(rankings, f)
f.close()
