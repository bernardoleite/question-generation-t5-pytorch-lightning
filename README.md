# question-generation-t5

### Todo list

- [ ] Translate Squad v1.1 to Portuguese
- [ ] Handle unexpected \n in dataset passages
- [ ] Handle paragraphs with > 512 tokens (after encoding)
- [ ] Provide appropriate names to experiences
- [ ] Save all (h)params in each experience
- [ ] Report BLUE 1-4 and RougeL evaluation metrics
- [ ] Figure out how to include additional evaluation metrics during training, validation and test
- [ ] Generate and decode during training, validation and test
- [ ] Call .test() after each epoch
- [ ] Understand the differences (and impact) of using .squeeze() and .flatten() 
- [x] Understand the differences between .generate() and forward() 
- [x] Save best model
- [x] Report logger results to .csv and tensorboard