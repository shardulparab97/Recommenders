import pickle
import math
import time

meanMovieRatings = {}
meanMovieCount = {}
userMovieMean={}
movies={}
baselines = {}
rankings={}
def loadMovieLens(path='./u.data'):
# Get movie titles

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

def transformPrefs(prefs):
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
# Flip item   and person
            result[item][person]=prefs[person][item]
    return result


#calculating the mean movie ratings
def getMeanMovieRatings(trainData):
    for user in trainData:
        for movie in trainData[user]:
            if movie not in meanMovieRatings:
                meanMovieRatings[movie] = 0
                meanMovieCount[movie] = 0
            meanMovieRatings[movie]+=trainData[user][movie]
            meanMovieCount[movie]+=1

    for movie in meanMovieRatings:
        meanMovieRatings[movie] /= meanMovieCount[movie]

#getting user movie mean values
def getUserMovieMean(trainData):
    for user in trainData:
        userMovieMean[user] = 0
        count = 0
        for movie in trainData[user]:
            userMovieMean[user] += trainData[user][movie]
            count += 1
        userMovieMean[user]/=count

#getting overall mean movie rating
def getOverallMeanMovieRating():
    count = 0
    overallMeanMovieRating=0
    for movie in meanMovieRatings:
        count+=1
        overallMeanMovieRating += meanMovieRatings[movie]
    overallMeanMovieRating/= count
    return overallMeanMovieRating

#getting all the baseline values
def getGlobalBaselineValues(userMovieMean,meanMovieRatings,overallMeanMovieRating):
    moviesAll = {}
    for line in open('../data/u.item'):
        (id,title)=line.split('|')[0:2]
        moviesAll[id]=title

    for user in trainData:
        baselines[user]={}
        for movie in movies:
            if movies[movie] not in trainData[user] and movies[movie] in meanMovieRatings:
             baselines[user][moviesAll[movie]]=userMovieMean[user] + meanMovieRatings[movies[movie]] - overallMeanMovieRating
            elif movies[movie] in trainData[user]:
                baselines[user][moviesAll[movie]]=trainData[user][moviesAll[movie]]


def transformPrefs(prefs):
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
# Flip item   and person
            result[item][person]=prefs[person][item]
    return result

#function for finding pearson similarity
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


# Gets recommendations for a person by using a weighted average
# of every other user's rankings
#get Recommendations based on User-User Collaborative Filtering
def getRecommendations(prefs,person,baseline,similarity=sim_pearson):
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
            #making use of baseline values
                totals[item]+=(prefs[other][item]-baselines[other][item])*sim
# Sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim
# Create the normalized list
    #rankings=[((total/simSums[item]),item) for item,total in totals.items( )]
    rankings[person]={}
    for item in totals:
    #rankings[person] = totals[item]/simSums[item]
        rankings[person][item]=baselines[person][item]+totals[item]/simSums[item]


trainData = loadMovieLens()
getMeanMovieRatings(trainData)

#print meanMovieRatings

getUserMovieMean(trainData)
#print userMovieMean


overallMeanMovieRating=getOverallMeanMovieRating()
#print overallMeanMovieRating

#print movies

getGlobalBaselineValues(userMovieMean,meanMovieRatings,overallMeanMovieRating)

#print baselinesinv

start = time.clock()
for i in range(1,944):
    print (i)
    getRecommendations(trainData,str(i),baselines)
end = time.clock()

print ("Time taken : ",(end-start))

f=open('./recommendations.p','w')
pickle.dump(rankings, f)
f.close()


