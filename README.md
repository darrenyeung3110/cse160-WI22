# cse160-WI22
Materials for CSE 160 Programming Assignments

## Setup Guide

## Get Access to DSMLP using VSCode:

Steps:

1. Install VS Code https://code.visualstudio.com/download
2. Install Remote-SSH plugin by searching for it in the extensions view
3. Click on the indicator on the bottom left corner

![image](https://user-images.githubusercontent.com/43923184/148268541-202b9806-7d08-415b-ad4d-7b4d04916388.png)

4. Click on Connect to Host.. and + Add new SSH Host...
5. Type in USERNAME@dsmlp-login.ucsd.edu (USERNAME to be replaced with your UCSD Active Directory username)
6. Click on where you want to save the SSH Configuration
7. Click on the Connect Popup
8. Type your UCSD Active Directory password when prompted and press enter
9. You are now connected to UCSD DSMLP! It can be verified by checking the bottom left corner which indicates dsmlp-login.ucsd.edu

A video is attached in case there are any issues with following the steps:

https://user-images.githubusercontent.com/43923184/148276847-f92fdbd4-14a4-4749-9b89-615c64b7ad81.mp4

## Access to CUDA and GPU:

Open terminal on VSCode when connected to DSMLP and run `launch.sh -g 1 -s -i ucsdets/nvcr-cuda:latest`
You should see an output like this:

![image](https://user-images.githubusercontent.com/43923184/148271105-200ed36c-dc88-4b01-9b68-cdb61a36b655.png)

This gives you access to GPU infrastructure on DSMLP; it starts a container with GPU access and loads it with a software image that contains CUDA and other basic packages. 

You must be within GPU container in order to properly compile. If you get an error about not having access to nvcc, then you are not in the container.

Please only use the container when you are compiling and release it when you are completed. 
