#----------------------------------------------------------------------------------------------------------------------------------------------
#' Author: Deivison V Souza
#' Institution: Federal University Pará
#' Analyze: Wood Species Identification using Convolutional Neural Network
#' DataSet: Macroscopic Images of Wood (Souza et al., 2020)
#' Manuscript: An automatic recognition system of Brazilian flora species based on textural features of macroscopic images of wood
#' Journal: Wood Science and Technology
#' Description: The dataset used in this code was used by Souza et al. (2020)
#' in the manuscript "An automatic recognition system of Brazilian flora species
#' based on textural features of macroscopic images of wood". The data available
#' at Mendeley Data (DOI: 10.17632/cc78ftcdf9.1).
#----------------------------------------------------------------------------------------------------------------------------------------------

# Begin -----

# 1: Install packages -------------------------------------------------------------------------------------------------------------------------
#install.packages("keras")
#install.packages("ggplot2")

# 2: Load packages into the current session ---------------------------------------------------------------------------------------------------
library(keras)
library(ggplot2)

# 3: Load a single image ----------------------------------------------------------------------------------------------------------------------
img <- image_load("dataset/Wood-Recognition/5-Bagassa guianensis/0526.jpg",
           grayscale = F, target_size = NULL)      # Load image

img <- img %>% image_to_array()                    # Convert to array
# dim(img)                                         # Array dimension (height,width,channels)
# length(img)                                      # Number of pixels (1530*2068*3)

# Channels
channel1 <- img[,,1]
channel2 <- img[,,2]
channel3 <- img[,,3]

# image_load("dataset/Wood-Recognition/5-Bagassa guianensis/0526.jpg",
#            grayscale = F, target_size = NULL) %>%
#   image_to_array() %>%
#   array_reshape(dim = dim(.)) %>%
#   as.raster(max = 255) %>%
#   plot()

# 4: Load dataset -----------------------------------------------------------------------------------------------------------------------------
path <- "dataset/Wood-Recognition"
img_width <- 2080
img_height <- 1540

image_generator <- image_data_generator(rescale=1/255, validation_split=0.2)

trainingSet <- flow_images_from_directory(
  directory = path,
  generator = image_generator,
  target_size = c(img_width, img_height),
  subset = "training",
  class_mode='categorical'
)

testSet <- flow_images_from_directory(
  directory = path,
  generator = image_generator,
  target_size = c(img_width, img_height),
  subset = "validation",
  class_mode='categorical'
)
