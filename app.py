import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import gensim
import datetime

app = Flask(__name__) #Initialize the flask App

#This function is used for the first search bar to search all JURIST articles
def all_articles(text):
    df = pd.read_csv('JuristData.csv')
    lower_text = text.lower()
    text_list = lower_text.split()
    temps = df[np.logical_and.reduce([df['4'].str.contains(word) for word in text_list])] 

    if len(temps) == 0:
        raise ValueError()
    else:
        pass
    
    combined = []
    date_range_text = []
    years = []

    for i in sorted(set(pd.DatetimeIndex(temps['0']).year)):
        years.append("  " + str(i)) #Displays each year on website
        temps['0'] = pd.to_datetime(temps['0'])
        year_data = temps[temps['0'].dt.year == i]
        
        string = []
        cleaned = []
        new_date = []
    
        for title in year_data['0']: #Used to clean the date and transform it into (Sep 18, 2010), for example.
            string.append(str(title))
        for word in string: 
            cleaned.append(word[:-15])
        for word in cleaned:
            new_date.append(datetime.datetime.strptime(word, '%Y-%m-%d').strftime('%b %d, %Y'))
        year_data['0'] = new_date

        dates = []
        for i in year_data['0']:
            date_dict = dict(date = i)
            dates.append(date_dict)

        first_date = dates[0]
        last_date = dates[-1]
        #Displays date range text on HTML
        date_range_text.append("• From " + str(first_date['date']) + " to " + str(last_date['date']))

        titles = []
        for i in year_data['1']:
            title_dict = dict(title = i)
            titles.append(title_dict)

        links = []
        for i in year_data['2']:
            link_dict = dict(link = i)
            links.append(link_dict)

        combined.append([{**d, **n, **l} for (d, n, l) in zip(dates, titles, links)]) #Information displayed on timeline
    massive_list = zip(years, date_range_text, combined) #Combines timeline information along with cluster information
    return(massive_list)

#This function is used for the second search bar to cluster articles
def model(text):
    df = pd.read_csv('JuristData.csv')
    texts = text.lower()
    temp = df[df['4'].str.contains(texts)]

    sent_vecs = {}
    for title in (temp['3']):
        sent_vecs.update({title: '0'}) #The zero is just used as a placeholder and doesn't mean anything
    sentences = list(sent_vecs.keys())

    vectors = []
    for i in temp['6']:
        i = np.array(np.matrix(i)).ravel()
        x = np.array(i)
        vectors.append(x)  

    #Number of clusters is determined by the number of articles based on the search term
    if len(temp) > 750:
        num_of_cluster = 10
    elif len(temp) > 550: 
        num_of_cluster = 9
    elif len(temp) > 400:
        num_of_cluster = 8
    elif len(temp) > 300:
        num_of_cluster = 7
    elif len(temp) > 200:
        num_of_cluster = 6
    elif len(temp) > 100:
        num_of_cluster = 5
    elif len(temp) > 50:
        num_of_cluster = 3
    else:
        num_of_cluster = 2

    kmeans = KMeans(n_clusters=num_of_cluster).fit(vectors)
    clustered_data = pd.DataFrame({'label': kmeans.labels_, 'sent': sentences})

    cluster_num = set(clustered_data['label'])
    empty = pd.DataFrame()

    for num in cluster_num:
        array = kmeans.transform(vectors)[:, num]
        index = np.argsort(array)[::][:10] #Gets 10 articles closest to the cluster center
        data = clustered_data[clustered_data['label'] == num]
        empty = empty.append(data[data.index.isin(index)])
    clustered_data = empty

    for value in set(clustered_data['label']):
        if len(clustered_data.loc[clustered_data['label'] == value]) < 5: #Removes clusters with less than 5 articles
            clustered_data = clustered_data[clustered_data['label'] != value]

    cluster_num = set(clustered_data['label'])
    
    if len(cluster_num) == 0:
        raise ValueError()
    else:
        pass
    
    combined = []
    number_text = []
    topics_text = []
    date_range_text = []

    for num in cluster_num:
        number_text.append('• Cluster ' + str(num)) #Display cluster number text for HTML
        example = clustered_data[clustered_data.label == num].sent.tolist()
        clustered = temp[temp['3'].isin(example)][['0', '1', '2', '5']]
        clustered['0'] = pd.to_datetime(clustered['0'])
        clustered = clustered.sort_values(by = '0') #Sorts clusters by date
        
        string = []
        cleaned = []
        new_date = []
        
        for title in clustered['0']: #Used to clean the date and transform it into (Sep 18, 2010), for example.
            string.append(str(title))
        for word in string: 
            cleaned.append(word[:-15])
        for word in cleaned:
            new_date.append(datetime.datetime.strptime(word, '%Y-%m-%d').strftime('%b %d, %Y'))
        clustered['0'] = new_date
        
        dates = []
        for i in clustered['0']:
            date_dict = dict(date = i)
            dates.append(date_dict)
     
        first_date = dates[0]
        last_date = dates[-1]
        #Displays date range text on HTML
        date_range_text.append("  " + "From " + str(first_date['date']) + " to " + str(last_date['date']))
        
        titles = []
        for i in clustered['1']:
            title_dict = dict(title = i)
            titles.append(title_dict)
        
        links = []
        for i in clustered['2']:
            link_dict = dict(link = i)
            links.append(link_dict)
        
        topic = []
        for lists in clustered['5']:
            topic.append(lists)
        topic_list = " ".join(topic)
        
        def word_count(str): #Used to count appearances of a word
            counts = dict()
            words = str.split()
            for word in words:
                if word in counts:
                    counts[word] += 1
                else:
                    counts[word] = 1
            return(counts)
        
        wordcount = word_count(topic_list)
        sort = sorted(wordcount, key=wordcount.get, reverse=True)[0:5] #Get the top 5 words by count of each cluster
        topics_text.append("  " + "Topics: " + str(sort))
     
        combined.append([{**d, **n, **l} for (d, n, l) in zip(dates, titles, links)]) #Information displayed on timeline
    massive_list = zip(number_text, topics_text, date_range_text, combined) #Combines timeline information along with cluster information
    return(massive_list)
 
@app.route('/')
def home():
    return render_template('webpage.html')

@app.route('/predict',methods=['POST', 'GET'])
def predict():
    int_features = request.form['experience']
    prediction = model(int_features)
    
    return render_template('webpage.html', items = prediction)

@app.route('/allstar',methods=['POST', 'GET'])
def article():
    int_features = request.form['articles']
    prediction = all_articles(int_features)
    
    return render_template('webpage.html', articles = prediction)

@app.errorhandler(500)
def page_not_found(e):
    return render_template('500.html'), 500

if __name__ == "__main__":
    app.run(debug=True)

