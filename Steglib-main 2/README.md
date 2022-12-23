# Steglib

A library for steganography

StegLib is a web application for accessing encryption and decryption steganalysis tools, with the click of a button.
Site is hosted with an Azure virtual machine, at this link: http://steglib.eastus.cloudapp.azure.com:5000/

## Inspiration:

Steganography is a method for concealing secret data within an ordinary file, such as an image, while avoiding suspicion and detection. Since 440 BC, steganography has been used to hide secret messages in ordinary objects, and have been used by governments, terrorists, and everyone in between. Steganalysis is such a useful but dangerous medium because it is a completely closed system; there is no way of knowing whether a message being sent is malicious or not. In recent decades, terrorist groups such as Al Qaeda have been transferring secret messages under the FBI’s nose without being detected for a long period of time! Our ability to detect steganography is severely underdeveloped, and this is a glaring issue for safety on the internet. Millions of images are passed over the internet, and most of them go completely unregulated and undetected. Thus, we decided to explore and improve current steganalysis technologies, in order to enable authorities and businesses to unveil secret messages ahead of time, to avoid the possibility of causing harm to our safety. We also wanted to educated and promote the importance of steganalysis, in order to promote future research in this subject, and foster internet safety. 

## What it does: 

Introducing StegLib, a encryption and decryption tool that allows anyone easy access to steganalysis image encryption and decryption. With the click of a button, anyone can check if a pertinent image has a secret message in it or not, and find out ways to detect them. We designed our website to be as simple as possible since we found that the market lacks steganalysis technology, and many don’t even know about the dangers of steganography. Our web application allows one to pass through large amounts of data and detect a secret message hidden well within the image. We want to raise people’s awareness towards steganalysis, and demonstrate how steganalysis works with visualizations and machine learning. Finally, StegLib improves on current state of the art steganalysis models, by giving a fast and efficient pipeline for steganalysis to be implemented, and reducing detection and decryption times for steganographed images. 

## How we built it:

For our front-end, we used basic HTML/CSS along with the bootstrap library to provide a clean and simple interface for users to access our Steganalysis. For the backend, we used flask to communicate with the frontend, and machine learning libraries such as Pytorch and Scipy to integrate machine learning. We also added algorithms using libraries like jpegio and albumentations for steganalysis decoding, leveraging methods such as least significant bit decryption, discrete cosine transform visualizations, and efficientnet CNN steganalysis. The full stack web application was deployed on an Azure virtual machine for easy access. Within our front-end team, we worked on providing the best user experience for users to allow them to easily access our steganalysis back-end technology. We went with a minimalistic design so that our website is just a simple tool that allows decryption and encryption to be done within less than 10 clicks. Additionally, we made sure the website flowed well and that it would be simple for users that are new to Steganalysis to understand our image manipulation process. For our machine learning model, we decided to use an Efficientnet B0 multiclass classifier ensemble, trained on the Alaska2 Image Steganalysis dataset, of images trained using several DCT steganalysis images, using the AUROC (area under the rate of change) metric. This helped us effectively calculate the percentage of an object being steganography using its DCT coefficients, as well as detect the most plausible steganography encrytpion scheme. DCT machine learning is a much stronger method than first order statistical attacks like visual attacks or least significant bit noise detection, which can be thwarted, but DCT is much harder to thwart using state of the art steganalysis methods. 

## Challenges we faced

This was our first time working with Bootstrap as a team for our frontend framework and had many issues aligning our content, since our styling had overlapped each other and was difficult at certain points. But we were able to overcome this effectively by using VSCode liveshare and communicating a design so everything looks flush. At first, we had some issues connecting the front end and back end. We solved these issues by integrating dotenv and flask sessions in order to save images and use them when needed. 
We had to do a lot of research into steganography in a short period of time, as it was the first time many of us were introduced to the concept of steganalysis. We had to do lots of research quickly to understand current state of the art steganalysis methods, and how to improve on them. We looked at papers for both steganalysis and machine learning, in order to figure out how to improve on steganalysis methods. We also had trouble figuring out our efficientnet model at first, so we had to go back to look at neural nets, and read Kaggle discussion and papers, and implemented albumentations in order to speed up the inferencing time. We also spent a long time looking for a good dataset in order to train our model, and we had to figure out a relevant train/test split for the model training. Finally, integrating the machine learning pipeline into flask was not that easy, as we had to use Flask sessions and PIL to streamline the DataLoader to take in an image, and spit out a prediction that could be sent back to the frontend. 

## Accomplishments that we're proud of:

We are proud of created a platform that enables users to easily access steganographic encrypting/decrypting tools through a user-friendly interface, and promotes learning about steganalysis. We learned about steganography and integrated machine learning principles into cybersecurity in a short period of time. We implemented and improved cutting edge steganalysis methods, and used many new libraries that we weren't familiar with before. We were able to do research and quickly write models and algorithms about a topic that we did not have much prior experience with. Finally, we were able to integrate the frontend and backend as a full stack pipeline, while leveraging Flask as a middleman. 

## What we learned:

We learned the concept of steganography and explored different methods/algorithms for encoding/decoding secret messages. It’s truly mind blowing to witness how these fascinating methods are invented and being applied so effectively in various places.We were blown away by the sheer amount of data, as well as the scope of improvement, for this topic. There is so much untapped potential for steganalysis, and although there is already research on it, there is still a lot of improvement possible on the subject. We hope to look into more steganalysis methods, and gain insight into state of the art steganalysis techniques, and how they work. We also obtained a more thorough understanding toward the bootstrap framework. Through using bootstrap for all of our pages, it makes our lives much easier without having to go through all the CSS. We learned a lot about using flask sessions and configurations in order to connect a full stack pipeline, and improve security. We implemented several methods to prevent attacks on our app (e.g. by adding a max file size). We learned about cybersecurity and its involvements in machine learning, as we learned about using infosec datasets to perform machine learning (using albumentations to perform multiclass efficientnet on stegangographed iamges). Last but not least, working as a team allowed us to learn a lot from each other! This hackathon was certainly an amazing experience. 

## What's next for StegLib:

We want steglib to become a service that can be used by businesses and governments, so we will make steglib into a standalone PyPi library and an API endpoint, so that users can easliy check if an image is suspicious or not. We will improve further on our machine learning models by training on more comprehensive datasets, and with trying out even more models, as well as doing more research on machine learning principles in steganalysis and cybersecurity. We also want to promote the importance of steganalysis and info security, so we want to add an educational game to steglib, to educate users about the importance of steganography and its cool factor, along with its dangers. 

References: 

https://www.giac.org/paper/gsec/3494/steganography-age-terrorism/102620

https://www.youtube.com/watch?v=TWEXCYQKyDc

https://www.aimspress.com/article/id/3687

https://www.wired.com/2001/02/bin-laden-steganography-master/

https://arxiv.org/abs/2111.12231

https://arxiv.org/abs/1912.06540
