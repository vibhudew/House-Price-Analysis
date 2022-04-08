# FDM-Mini-Project

# House Data Predictor

#### a. Dataset
<p>The reason behind why we select the house dataset because in current world house construction companies are in a difficult situation when they are estimating the relevant information. This data set contains house sale prices for king country, which includes Seattle. It includes homes sold between May 2014 and May 2015. Data set consists of 21614 rows and 21 columns.

<a href="https://www.kaggle.com/datasets/harlfoxem/housesalesprediction?select=kc_house_data.csv">Click to View Dataset</a>
<br>

#### b. Problem Identification
<p>‘Homelands Constructions’ is the best-known house constructing 
company in USA. After they constructed the house, based on the requirements, they need to 
predict the type of the house and after before building a house also they need to predict the number 
of floors that house should have based on requirements. And they need to solve the issues that 
happens when pricing houses. Because the houses with same facilities may have different prices. 
Solving these issues can be very time consuming and exhaustive.</p>

#### c. Solution
<p>We plan to build a K means clustering model to identify the type of the house. The goal of clustering is to determine the intrinsic grouping in a set of unlabeled data. And it is very cost effective and easy to build. Also we plan to build a linear regression model to predict the number of floors based on the requirements and Gradient Boosting algorithm to predict the house price for already built house.These predictions of the house varying on no of rooms in the house, no of bathrooms ,square foot of area in different places in the house ,etc. Through the models which we are designing they can simply enter the details and sort out the predictions of house. And by this when customer enter details of a new house, they can predict the house condition. First, we will be applying necessary pre-processes to data set as mentioned in below sections. Then we built necessary models to predict details. After that model validation will be done. And next we plan to build the web application where users can apply the model and predict details.
</p><br>

<h1><a href="https://homelandspredictor.herokuapp.com/">Click to See the Project</a></h1>

## Note

### Since this is an academic data mining project , we have focused more on data mining techniques, exploratory data analysis(EDA) ,modeling and accuracy of predictions other than front-end development.

<br>

## Tools and Technologies used
- Python 3 <p align="center"><a href="https://www.python.org/doc/" target="_blank"><img src="https://www.pngitem.com/pimgs/m/31-312064_programming-icon-png-python-logo-512-transparent-png.png" width="130"></a></p>
- Python Flask <p align="center"><a href="https://flask.palletsprojects.com/en/2.0.x/" target="_blank"><img src="https://miro.medium.com/max/876/1*0G5zu7CnXdMT9pGbYUTQLQ.png" width="180"></a></p>
- Heroku <p align="center"><a href="https://devcenter.heroku.com/" target="_blank"><img src="https://dailysmarty-production.s3.amazonaws.com/uploads/post/img/509/feature_thumb_heroku-logo.jpg" width="120"></a></p>
- Front-end Development
<table align="center">
  <tr>
    <td><a href="https://html.spec.whatwg.org/" target="_blank"><img src="https://img.icons8.com/color/200/html-5--v1.png" width="180"></a></td>
    <td><a href="https://www.w3.org/TR/CSS/#css" target="_blank"><img src="https://img.icons8.com/color/200/css3.png" width="180"></a></td>
    <td><a href="https://getbootstrap.com/docs/5.0/getting-started/introduction/" target="_blank"><img src="https://img.icons8.com/color/200/bootstrap.png" width="180"></a></td>
  </tr>
</table>


## Acknowledgment
<p align="center"> This is a project done for the Fundamentals of Data Mining (IT3051) of BSc.(Hons.) Degree in Information Technology Specialising in Data Science in Sri Lanka Institute of Information Technology by three team members</p>
<p align="center"> <a href="https://www.sliit.lk/" target="_blank"><img src="https://upload.wikimedia.org/wikipedia/en/a/a6/SLIIT_Logo_Crest.png" width="100"></a></p>


