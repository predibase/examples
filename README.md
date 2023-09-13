<p align="center">
  <a href="https://predibase.com/">
    <img src="misc/images/logo.png">
  </a>
</p>

# Predibase Examples

<br/>

## Basic Examples

Here we provide a few examples to get you started with Predibase. These examples are designed to be easy to follow and
provide a quick introduction to the main features of Predibase.

<br/>

## Model Augmented Generation

In this example, we train a recommender system model to predict the likelihood of a user purchasing a product based on
their past product interactions. We then pass the output of this model to a Predibase hosted LLM to generate 
personalized email outreach campaigns specific to each user. This example will show you both how to train the
recommender system model and how to chain the outputs with Predibase LLM capabilities.

<br/>

## Information Extraction / RAG

In this example, we parse a set of documents into chunks of text and then perform information extraction with Retrieval
Augmented Generation (RAG). This directory contains a Jupyter notebook with the full example tutorial in addition to 
utility functions supporting the example Streamlit application.

<br/>

## Supervised ML Web Application

This example shows how to use Predibase to build a web application that is backed by a supervised ML model for make 
predictions. Specifically, the application predicts a passenger's likelihood of surviving the Titanic disaster based on
a variety of input features.