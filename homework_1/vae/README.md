# Variational Autoencoders: MNIST experiments

So here i did some experiments with Autoencoders, VAE and their variants (f-VAE, $\beta$-VAE). Quick summary of what happened.

### 1. Autoencoder & compression
Started with regular Autoencoder to see how much i can compress the digits.
- Tried latent sizes: 2, 4, 8, 16, 32.
- **What i got:** With size 2 the digits look like blobs, cant really tell what they are. With 8+ its pretty good, almost perfect. 

### 2. VAE and the blurry problem
For VAE i first used simple **fully connected** architecture (MLP) like in the Autoencoder.

**Result:** Generated images were super blurry. Model kinda captured the general shape of digit but all details were gone. Edges between digit and background were unclear.

**Solution:** Switched to **convolutional** architecture (`Conv2d` for encoder, `ConvTranspose2d` for decoder).
This made huge difference. Convolutions helped the model keep spatial structure and generations became much sharper.

### 3. f-VAE (Flow-based VAE)
Plugged in the RealNVP model (from previous task but 1D version) into the latent space of VAE.
**Result:** It gave more flexible posterior. Loss dropped a bit and samples seemed a bit more diverse compared to regular Gaussian prior, tho visually the difference wasnt that big on simple dataset like MNIST.

### 4. Bonus: $\beta$-VAE
Experimented with $\beta$ parameter to control KL divergence weight.
- **$\beta = 4.0$**: Latent space became really structured. Moving one dimension changed **rotation**, another changed **thickness**. Disentanglement worked!
- **$\beta = 0.5$**: Better reconstruction (sharper images) but latent dimensions got entangled again.

## Conclusion
Compared to Normalizing Flow task this was way easier. These models trained in minutes while RealNVP took hours

No exploding gradients or NaN errors. VAEs are pretty robust. Much easier to implement and debug
