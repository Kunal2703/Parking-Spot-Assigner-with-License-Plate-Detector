
# Parking Spot Assigner with License Plate Detector
## 1. INTRODUCTION
Nowadays, people don't use much public transportation because it is a bit time-consuming. But when every individual thinks the same and travels in their own vehicle, it leads to more traffic jams, time consumption also due to this there is a lot of honking and leads to noise pollution near an educational institute. 
This is the same case which we noticed at the entry gate of our college. Due to the clotting of the vehicles at the entry gate, there is a lot of chaos and commotion. The process of manually registering the car number and allotting the parking slots is a bit time-consuming and requires manpower as one has to note down the car number and the other would provide the tokens for the parking. This assignment of the tokens is also done randomly, which is also a hectic task if the parking provided is far from that employee seating. So, in this present scenario, there is no specific parking for any department.  
To overcome all these problems, we propose an automated robust system to reduce the waiting time by scanning the vehicle's number to identify the details of the employee and assign the parking slot near to their office space on a first come first serve basis. 

### 1.1 Purpose of the Project 
 
The purpose of this project is to reduce the time-consumption caused due to the manual process of noting down number plate by automating the process completely. The project also aims at making the slot allotment process less random by allotting an empty slot which nearest to the employees seating.  
 
### 1.2 Target Beneficiary 
 
The target beneficiaries of the proposed methodology are the employees who commute in their own cars as they can enter into the college smoothly without waiting in long queues, the guards as their workload of noting down details of the vehicle will reduce, and everyone who is near the college as there will be less commotion which will help make the environment more peaceful. 
 
### 1.3 Project Scope 
 	 
The scope of the project is to reduce the workload of the guards, to save the time of the employees entering the college premises via their cars and effectively allotting parking slots based on the location of their seating. 


## 2.	PROJECT DESCRIPTION
### 2.1 Data/ Data Structure 
 
The dataset used for training the machine learning model consists of images, both the front view and rear view of a car.  The dataset also consists of labels that provide coordinates of the bounding box around the number plate. The labels provide information about the center x and y coordinates, and the height and width of the bounding box. The dataset has been split according to a 70-20-10 split which is used for training, validation, and testing respectively. The dataset is made up of two separate datasets acquired from Kaggle. The first dataset is taken from Car License Plate Detection. The second dataset is taken from the Labeled licence plates dataset. Flip, crop, and shear augmentations have been applied to each image. 

### 2.2	Project Features 
 
The main feature of this project is to automatically determine the location of the number plate using machine learning and perform number plate text detection using OCR. Then the detected number plate will be used to assign the parking slot near the seating of the faculty. This will reduce manual dependence by automating the allotment process. It will manage the information of vehicles and parking slots and track all the information of Vehicles, Employees, Duration etc. 
A.	Determining the bounding box in object  
B.	Pre- Processing 
    a.	Cropping of number plates 
    b.	Resizing the image 
    c.	Gray scaling 
    d.	Removing noise 
    e.	Thresholding 
C.	Applying OCR  
 
### 2.3	Design and Implementation Constraints 
 
•	Requirement of GPU 
•	High resolution cameras required 
•	Difficulty in detection due to unidentifiable font 
•	Requirement of a custom dataset  

### 2.4	Design diagram
<img width="451" alt="image" src="https://user-images.githubusercontent.com/78562069/210048107-2bc6b9f8-5945-4104-8c4e-df1c12171fca.png">
      Fig1- Working of Vehicle Management System

<img width="451" alt="image" src="https://user-images.githubusercontent.com/78562069/210048133-32d94561-727e-4c0c-9609-e865b24c0db6.png">
      Fig 2 – Training YOLO Model
     
     
<img width="451" alt="image" src="https://user-images.githubusercontent.com/78562069/210048173-45824374-7b6f-4e72-a0e4-e47d2ab9ed92.png">
      Fig 3- Output of YOLO model
  
  
<img width="451" alt="image" src="https://user-images.githubusercontent.com/78562069/210048206-754af53c-7b84-4891-9c59-269c3a07b08d.png">
      Fig 4- Workflow of Tesseract OCR
      
 
## 3.	SYSTEM REQUIREMENTS 
 
Two separate computers with GPU and camera connected to them and a database to store details of employees, parking slots and the entered vehicles. 

## 4.	NON-FUNCTIONAL REQUIREMENTS 
4.1	Performance requirements 
 
The following are some basic requirements needed: 
•	Reliability 
•	Robustness 
•	Usability 

## 5.	IMPLEMENTATION 
 
When the car arrives in front of the entry gate camera, YOLO identifies the position of the numberplate and crops the image. The image is then resized and preprocessed before it is passed to tesseract OCR to extract the text. The extracted text is then checked with regular expression of the standard Indian numberplate. If it is a match, then the numberplate text is further checked with the database. First, we check if the numberplate has already entered to avoid duplicate entries. If the numberplate has not entered, then we check it with the employee table to see if the vehicle belongs to the employee. Once checked, we assign the available parking slot to it.  

<img width="524" alt="image" src="https://user-images.githubusercontent.com/78562069/210050085-702ad0cd-d906-4ced-9e4a-7b4ac011fe59.png">
fig 5- Code Snippet.

<img width="260" alt="image" src="https://user-images.githubusercontent.com/78562069/210050166-952aa7de-d73f-48ae-85ce-35165a33981d.png">
fig 6- YOLO identifies the numberplate.

<img width="467" alt="image" src="https://user-images.githubusercontent.com/78562069/210050197-519fac6b-05d4-4aac-a02c-c4e61d7a0734.png">
fig 7- Tesseract OCR identifies the numberplate text and assigns slot.

<img width="179" alt="image" src="https://user-images.githubusercontent.com/78562069/210050230-67469269-08e9-4db1-9a5a-ceb8095c6ffb.png">
fig 8- The numberplate is entered in the entry table.

When the car arrives in front of the exit gate camera, again the same process takes place. The numberplate text extracted is first checked with entry table. If it is present in the table, the slot assigned to that vehicle is released and the numberplate record is removed from the entry table. 

<img width="482" alt="image" src="https://user-images.githubusercontent.com/78562069/210050304-c1bdad52-9c36-4866-a801-6812a30e1322.png">
fig 9- Numberplate is identified and the assigned slot is released.

