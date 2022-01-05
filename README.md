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

https://user-images.githubusercontent.com/43923184/148270493-460d2b8d-b344-45f0-bc2a-6d5d75ded223.mp4

## Access to CUDA and GPU:

Open terminal on VSCode when connected to DSMLP and run `launch.sh -g 1 -s -i ucsdets/nvcr-cuda:latest`
You should see an output like this:

![image](https://user-images.githubusercontent.com/43923184/148271105-200ed36c-dc88-4b01-9b68-cdb61a36b655.png)

This gives you access to GPU infrastructure on DSMLP.

## Build the libgputk Library

We use libgputk which contains the library for cuda. To build this library navigate to the `libgputk` folder and run the following commands:
1. `make` 
2. `make libgputk.a`

This completes the setup process! Instructions for each lab is under the specific Lab folders.
