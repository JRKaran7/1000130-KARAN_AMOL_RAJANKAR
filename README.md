# Developing a Computer Vision-Based Detection System using Machine Learning
## Faint Detection System


Karan Amol Rajankar </br>
Delhi Public School – Bangalore North </br>
Email ID: - karan.rajankar07@gmail.com </br>

#### Introduction
“Computer vision and machine learning have really started to take off, but for most people, the whole idea of what is a computer seeing when it's looking at an image is relatively obscure.” – Mike Krieger </br>
Computer Vision is one of the booming technologies that AI engineers are working upon. It is used in autonomous vehicles, recognition systems, and robotics to provide computers the eyes that they need for detecting entities. Combining human accident occurrences with computer vision can help detect and immediately call the necessary authorities, therefore making sure that the person receives aid on time. This report will be covering an application of computer vision that can detect a person fainting and sending a message to the closest family member (considering that twilio requires a fee to pay for calls, we have used Whatsapp as a mode of communication for our project).

#### Problem Statement
Syncope is a type of condition where the blood flow slows down, leading to in home fainting. A lot of times people stay alone and unfortunately if they faint, no one will realise, leading to late medications. The statistics below shows that how the syncope rate for adults is more, especially the older population. </br> 
![image](https://github.com/user-attachments/assets/992f35de-7db6-4e0f-8f09-d1aec43bc269)

About 30 – 40% of adults experience a fall related injury in their homes. Not only this, about 35 – 40% of fainting episodes lead to injuries that require immediate medical attention. </br>
Looking at the statistics, I feel that this issue must be taken into deep consideration and addressed. Also, while addressing the issue, conditions like accuracy and quicker messaging must also be taken into consideration.

#### Goal and Objective
To develop a system can, detect a person fainting and immediately inform the closest family member.
#### Approach
Here is how the Software Development Life Cycle (SDLC) helped me create the faint detection system: - </br>
#### 1.	Planning: - 
•	Having discussed this with my teachers and family members, I found that there are a higher rate of injuries due to falling in homes and fainting that needs to be resolved. </br>
•	So, I thought of using computer vision to make a faint detection system that can detect three different poses: sitting, standing, and fainting. Along with this, when fainting is detection, it will WhatsApp message the closest family member. <br>
•	I came up with this idea because my grandparents had recently met with two fainting episodes that my family feels need to be detected immediately and provide them medications before it becomes too late. <br>
•	I generated a base problem statement with statistics, which will help me work on my project. <br>
•	I first thought of detecting only fainting, but I then decided to even detect sitting and standing so that the accuracy of detecting faint increases. <br>
•	Existing projects accessed from ResearchGate and IEEE Xplore gave me an idea of how the fainting detection system functions. <br>
 
#### 2.	Defining: -
•	Required Python computer vision, media pipe and WhatsApp message coding experience. I learned about it through many YouTube videos and websites like GeeksForGeeks, Java Tutorials, etc. <br>
•	I used websites like vector images and Shutterstock to download different sitting, standing, and fainting videos. Added 15 videos for each pose to increase accuracy. <br>
•	I asked ChatGPT to provide the different points of the body to detect three different poses. <br>
•	Finally, understood the use of pywhatkit for the code to send the WhatsApp message to the provided number. This will be the final output of the code. <br>
 
#### 3.	Designing: - 
•	I began designing the faint detection. <br>
•	It is a working detection system that checks three different poses. If fainting is detected, then it sends a WhatsApp message to the given number saying that fainting has been detected. <br>
•	I designed a storyboard to express how this detection system can be beneficial and a requirement for all houses. <br>
•	Finally, I designed a flowchart to show the system process from the start to the end. <br>

#### 3.	Building: - 
•	I started building my faint detection system on Visual Studio Code. <br>
•	Used modules like media pipe, pywhatkit, cv2, and sklearn to extract features from different training videos, trained a Decision Tree Classifier model to increase the accuracy of the system, and send an immediate WhatsApp message. <br>
•	I have used cv2 in order to capture real time frames and make this system functional real time. <br>
 
#### 4.	Testing: -
•	I tested my system by introducing it to my peers, teachers, and family members. <br>
•	They found the idea and the system quite interesting, and they all agreed that the system has a greater scope in the future. <br>
•	Introduced a feedback survey to make sure that the system was fulfilling the objective. <br>
 
#### 5.	Deployment: - 
•	Presented the faint detection system code in a GitHub repository to understand the code. <br>
•	Along with this ReadMe file includes screenshots of the output and the frame window. <br>

#### Future Scope
This system can be made much better by introducing it in the public surveillance cameras and introduce vital systems so that the camera provides the recorded vitals when the person had fainted. Along with this, immediate ambulance services and increased accuracy can make the project better.
#### Limitations
•	Can only send a WhatsApp message as Twilio requires a subscription for calling services <br>
•	Not much detailed feature extraction and limited to only one person at a time

#### Conclusion
While coding this faint detection system, I learnt about how Python can be used to code computer vision application to create systems for fulfilling basic necessities in the society. I also learnt about how to provide a proper dataset as all the videos were put in a list. So, the best videos were found by surfing through the internet. Finally, the discussion with peers, family members, and my teachers not only developed my communication skills, but they also gave me ideas and suggestions on how I can make my faint detection system better.
