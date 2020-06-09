#Creating web application by using frontend html and backend python
#Step 1:import lib
from flask import render_template, Flask, request,url_for
from keras.models import load_model
import pickle 
import tensorflow as tf
graph = tf.get_default_graph()
with open(r'CountVectorizer','rb') as file:
    cv=pickle.load(file)
cla = load_model('comment_text1.h5')
app=Flask(__name__)#interface btw server and my appln wsgi(web server gate interface helps to link)
#routing the appln
@app.route('/')#/ is used to bind the url
def helloworld():
    return render_template("page2.html")
@app.route('/login', methods= ['GET','POST'])#/ is used to bind the url
def admin():
    p=request.form["comment"]
    topic=cv.transform([p])
    print("\n"+str(topic.shape)+"\n")
    with graph.as_default():
        y_pred = cla.predict(topic)
    print("pred is "+str(y_pred))
    if(y_pred > 0.5):
        topic = " toxic"
    elif(y_pred < 0.6):
        topic = " non-Toxic"
       
    
    return render_template("page2.html", y="the comment is: "+str(topic))
  
if __name__=='__main__' :
    app.run(debug=True)
    