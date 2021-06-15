using AstroImages
using Flux
using DataLoaders
using Images
using DataFrames, CSV
#import LearnBase: nobs, getobs
import LearnBase
#Using DataLoaders' example.
##
#Dataset struct, holds filenames, the dir_path, IDs, and classes.
struct ImageDataset
    files::Vector{String}
    dir::String
    IDs::Vector{Int64}
    classes::Vector{Int64}
end

#Gets the ID from a filename.
get_id(ss::String) = parse(Int64,(split(split(ss,"-")[2],".")[1]))
const data_dir = "Data/"

#Constructor.
function ImageDataset(data_dir::String, classification_dir::String)
  ImageDataset(readdir(data_dir),data_dir,get_id.(readdir(data_dir)),
  DataFrame(CSV.File(classification_dir)).is_lens)
end
data = ImageDataset(data_dir,"classifications2.csv")

function LearnBase.nobs(ds::ImageDataset)
  length(ds.IDs)
end

function LearnBase.getobs(dataset::ImageDataset, i::Int)
  subpath = dataset.files[i]
  file = joinpath(dataset.dir, subpath)
  data = reshape(AstroImage(file).data[1],101*101)
  label = dataset.classes[i]
  data, label
end
give_data(ai::AstroImage) = reshape(ai.data[1],10201)

#Having a getobs(dataset::ImageDataset, i) fails, so I wrote each version.
function LearnBase.getobs(dataset::ImageDataset, range::UnitRange{Int64})
  subpath = dataset.files[range]
  file = joinpath.(dataset.dir, subpath)
  data = hcat(give_data.(AstroImage.(file))...)
  #data = file
  label = dataset.classes[range]
  data, label
end

dataloader = DataLoaders.DataLoader(data, 10; collate = false, partial = false)
#dataloader = DataLoaders.DataLoader(data, 10)
##
using Flux: train!
f = Dense(101*101,1)
train = reshape(LearnBase.getobs(data,10)[1],101^2)
label = LearnBase.getobs(data,10)[2][1]
f(reshape(train,101*101))
loss(x,y) = Flux.mae(f(x),y)
loss(train,label)
opt = Descent()
parameters = params(f)

trains, labels = LearnBase.getobs(data,1:10)
#train!(loss, parameters, zip(trains,labels), opt)
train!(loss, parameters, dataloader, opt)

println("It ran")
