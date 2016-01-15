module FeaturesProvider
open System
open System.IO
open System.Text.RegularExpressions
open System.Collections.Generic
open DataAccess
open NeuThink.Neuron
open NeuThink.NeuronTraining
open NeuThink.NeuralLayers
open NeuThink.DataSources



//MIC CODE

let mutable moc_classname = "worth"
let mutable moc_datname = "text"


let N (data:structured_enitity_provider) =
 (float) data.Entries.Length

let Nx (data:structured_enitity_provider) term classname =
 let dt = data.Entries
 let dclass = dt |> Array.filter (fun x -> ((x.[moc_datname].ToLower()).Contains(term)))
 (float) dclass.Length

//number of docs in class that contains term
let N11 (data:structured_enitity_provider) term classname =
 let dt = data.Entries
 let dclass = dt |> Array.filter (fun x -> (x.[moc_classname] = classname) && ((x.[moc_datname].ToLower()).Contains(term)))
 (float) dclass.Length + 1.0

//number of docs in not class that  contains term
let N10 (data:structured_enitity_provider) term classname =
  let dt = data.Entries
  let dclass = dt |> Array.filter (fun x -> (x.[moc_classname] <> classname) &&  ((x.[moc_datname].ToLower()).Contains(term)))
  (float) dclass.Length + 1.0 

//number of docs not in class that contains term
let N01 (data:structured_enitity_provider) term classname =
  let dt = data.Entries
  let dclass = dt |> Array.filter (fun x -> (x.[moc_classname] = classname) && not ((x.[moc_datname].ToLower()).Contains(term)))
  (float) dclass.Length + 1.0
  
//number of docs not in class that don't contain term
let N00 (data:structured_enitity_provider) term classname =
  let dt = data.Entries
  let dclass = dt |> Array.filter (fun x -> (x.[moc_classname] <> classname) && not ((x.[moc_datname].ToLower()).Contains(term)))
  (float) dclass.Length + 1.0

let N1p (data:structured_enitity_provider) term classname =
  (N11 data term classname) + (N10 data term classname)

let Np1 (data:structured_enitity_provider) term classname =
  (N01 data term classname) + (N11 data term classname)
  
let N0p (data:structured_enitity_provider) term classname =
  (N01 data term classname) + (N00 data term classname)

let Np0 (data:structured_enitity_provider) term classname =
  (N00 data term classname) + (N10 data term classname)

let MIC (data:structured_enitity_provider) term classname =
   let n11 = N11 data term classname
   let n1p = N1p data term classname
   let n01 = N01 data term classname
   let n10 = N10 data term classname
   let n0p = N0p data term classname
   let np1 = Np1 data term classname
   let np0 = Np0 data term classname
   let n00 = N00 data term classname
   let nx = Nx data term classname
   
   let n = N data
 //  Console.WriteLine((n*n11/(n1p*np1)).ToString())
   
   (n11/n) * Math.Log(n*n11/(n1p*np1),2.0) + (n01/n)*Math.Log(n*n01/(n0p*np1),2.0) +
   (n10/n) * Math.Log(n*n10/(n1p*np0),2.0) + (n00/n)*Math.Log(n*n00/(n0p*np0),2.0)

//END MIC CODE


let mic_words (data:structured_enitity_provider) dict classname =
 let words_freq = Array.map (fun x -> ((MIC data x classname),x)) dict
 let sorted = words_freq |> Array.sortBy (fun x -> fst x) 
 sorted
 //let srt = sorted |> Array.map (fun (x,y) -> x.ToString() + " " + y)
 //let vocs = sorted |>  Array.map (fun (x,y) -> y)


let filter_dict in_file (data:structured_enitity_provider) thrsld =
 let writen_data = new ResizeArray<string>()
 let words = new ResizeArray<string>()
 let dictload = File.ReadAllLines in_file
 Console.WriteLine(moc_classname)
 let classes = data.getClasses(moc_classname)
 for tclass in classes do
  if tclass<>"other" then
   writen_data.Add("class = " + tclass)
   Console.WriteLine("Analyzing " + tclass)
   let mics_for_class = mic_words data dictload tclass
   mics_for_class |> Array.map (fun (x,y) -> x.ToString() + " " + y) |> Array.iter (fun x -> writen_data.Add(x))
   mics_for_class |> Array.filter (fun (x,y) -> x > thrsld) |> Array.map (fun (x,y) -> y) |> Array.iter (fun x -> words.Add(x))
 File.WriteAllLines(moc_classname+".dta",Array.ofSeq writen_data)
 let wrds = words |> Array.ofSeq  |> set |> Array.ofSeq 
 File.WriteAllLines(moc_classname+".voc",wrds)
 






let fp_is_number line = 
   let pp = Regex.Matches(line,"[\\d]{3,6}(\.[\d]0){0,1}")
   (pp.Count=1) && (pp.[0].Index=0) && (pp.[0].Length = line.Length) 



type features_provider(entries:structured_enitity_provider,target_class:string) = 
 let mutable feats = [||]
 let mutable features_list = [||]
 
 let mutable generator_function = (fun  (entry:Dictionary<string,string> array) (i:int) -> features_provider.provide_basic_features_train target_class entry.[i])

 let mutable continous_generator_function =  (fun  (entry:ResizeArray<Dictionary<string,string>>) (i:int) -> [|0.0;0.0|])

 let mutable gen_type = "discrete"

 let mutable classes = entries.getClasses(target_class)

 //member this.compute_features () =
 // entries.Entries |> Array.map (fun x -> Array.append (this.provide_basic_features_train x) [|x.[target_class]|])
 
 
 
 let unroll_features features all_features =
 // start_profile "features_transform"
  let p = Array.map (fun x -> if (Array.exists (fun y ->  x=y) features) then x else "#no" + x  ) all_features
  //end_profile "features_transform"
  p

 member this.DataSet = entries
 member this.Classes with get() = classes and set x = classes <- x

 member this.FeaturesType with get() = gen_type and set x = gen_type <- x

 member this.digital_output i =
  classes |> Array.map (fun x -> if  x = (entries.getEntry i).[target_class] then 1.0 else 0.0)
 
 member this.digital2class (digital: float array) =
  let mutable max = 0.0
  let mutable index = 0
  for i=0 to digital.Length - 1 do
   if digital.[i] > max then
    max <- digital.[i]
    index <- i
 // Console.WriteLine(index)
 // Console.WriteLine(classes.[index])  
  classes.[index]


 member this.sample_class (digital: float array) =
  let p = digital |> Array.mapi (fun i x -> (x,i))
  let sorted = p |> Array.sortBy (fun (x,i) -> 0.0 - x)
  let rnd = new System.Random()
  let u = rnd.Next(2)
  let classx = sorted.[u]
  classes.[(snd classx)]


 member this.select_words source thrsld =
   Console.WriteLine("Generating all-word features...")
   this.generate_features_list ()
   this.save_features_list ()
   moc_classname <- target_class
   moc_datname <- source
   Console.WriteLine("Selecting word features...")
   filter_dict (target_class+".fea") entries thrsld
 
 member this.features with get() = feats

 member this.generator with set(x) = generator_function  <- x

 member this.generator_continuos with set(x) =  continous_generator_function  <- x
 
 member this.features_array () =
   entries.Entries |> Array.mapi (fun i x -> generator_function entries.Entries i)
 
 member this.compute_features(i) = 
   unroll_features (generator_function entries.Entries i) features_list
   
 member this.compute_features_digital(i) = 
  let p = unroll_features (generator_function entries.Entries i) features_list
  p |> Array.map (fun x -> if x.Contains("#no") then 0.0 else 1.0)
  
 member this.Indexes () =
  let p = new ResizeArray<int>()
  for i = 0 to  entries.Entries.Length - 1 do
   let e = entries.getEntry i
   p.Add((int) (e.["index"]))
  Array.ofSeq p


 member this.compute_features_direct(i)  =
   continous_generator_function entries.EntriesDirect i


 member this.compute_features_uni(i) =
   if this.FeaturesType = "discrete" then
     this.compute_features_digital(i)
   else
     this.compute_features_direct(i)

 member this.Count() =
  if this.FeaturesType = "discrete" then
     this.compute_features_digital(0).Length
  else
     this.compute_features_direct(0).Length
 
 member this.compute_all_features_digital() =
  Console.WriteLine("computing features...")
  if this.FeaturesType = "discrete" then
    entries.Entries |> Array.mapi (fun i x -> this.compute_features_digital(i))
  else
    entries.Entries |> Array.mapi (fun i x -> this.compute_features_direct(i))
 
 member this.all_digital_output() =
    Console.WriteLine("computing results vector...")
    //{for x in entries.Entries -> this.digital_output i}
    entries.Entries |> Array.mapi (fun i x -> this.digital_output i)
 
 member this.compute_all_features ()  =
  let generator_func = generator_function

  let features = Array.map (fun x -> unroll_features x features_list) (this.features_array ())

  let result = Array.map2 (fun (x:string array) (y:Dictionary<string,string>) -> Array.append x [|y.[target_class]|])  features entries.Entries
  feats <- result
  result
 
 member this.compute_features_basic() =
  this.compute_all_features ()
  this.save_features (target_class + ".trn")
 
 
 member this.save_features filename =
  File.WriteAllLines(filename, (Array.map (fun x -> String.concat " " x) feats))
 
 static member provide_basic_features_train target_class (entry:Dictionary<string,string>) =
  let features = new ResizeArray<string>() 
  for line in entry do
   if line.Key <> target_class then
    let data = line.Value
    let newdata = data.Split([|" "|],StringSplitOptions.RemoveEmptyEntries)
    for d in newdata do
     if (fp_is_number d) then
      features.Add("sp_number")
     else
     features.Add(d)
  Array.ofSeq features  
 
 //Interface functions 
 member this.Item with get(i) = this.compute_features(i)

 member this.Entries with get() = entries.EntriesDirect

 member this.load_features_list() =
  if this.FeaturesType = "discrete" then
   features_list <- File.ReadAllLines(target_class + ".fea")
  else
   features_list <- [||]

 member this.save_features_list () =
  File.WriteAllLines(target_class + ".fea",features_list)

 member this.generate_features_list () =
  features_list <- (this.features_array ()) |> Array.fold (fun x y -> Array.append x y) [||] |> Set.ofArray |> Set.toArray





type elman_trainer (features:features_provider,target_class:string,network:ITrainableNetwork,cv:(features_provider*(features_provider->string->ITrainableNetwork->unit)) option) =

 let train_model =
  if features.FeaturesType <> "continous" then
   features.generate_features_list ()
   features.save_features_list ()


  let input_size = features.Count()
  let output_size = (features.Classes).Length
  Console.WriteLine("input size= " +  (input_size.ToString()))
  Console.WriteLine("output size= " +  (output_size.ToString()))

  let features_data = features.compute_all_features_digital()
  let ys = features.all_digital_output()

 //let network = new BRNN_Elman(25,input_size,output_size,false)

  //network.Load (mydir + target_class + ".mod")
  //network.DeUnDropOut()
 //hidden (input) layer
 // Console.WriteLine("Creating network")
  //let network = new RCNN(input_size,input_size,output_size)
  //network.AddNoiseLayer(input_size,input_size,1,true,0.01)
 // let proj_layer = network.AddProjectionLayer((wordVectors.base_dictionary.Count)+1,3,50,1,true)



  //network.FinalizeNet()
  network.Save(target_class + ".mod")
  File.WriteAllLines(target_class+".cls",features.Classes)


  //network.AddDropOutLayer(50,50,3,false)
  //network.AddRecurrentLayer(20,53,2,false)
  //network.AddDropOutLayer(15,15,2,false)
  //network.AddTanhLayer(50,50,2,false,activation_relU )

  //network.AddDropOutLayer(50,50,3,false)

  //init projections
  //for i = 0 to wordVectors.projections.Length - 1 do
  // proj_layer.Projections.[i] <- wordVectors.projections.[i]

  //network.Load (target_class + ".mod")
 //gradient_descent_perceptron_online  80 network inps outs
  Console.WriteLine("Starting RPROP...")
  let indexes = features.Indexes()
  Console.WriteLine("Starting RPROP...2")
  //BRNN_SGD_rmsprop  30 network  (new SimpleProvider(features_data))  (new SimpleProvider(ys)) indexes
  //rprop_uni_bptt
  match cv with
   | None -> (RmspropSGD 20 network (new SimpleProvider(features_data)) (new SimpleProvider(ys)) indexes 0.0001 None)
   | Some(cvx,valfunc) ->  (MomentumSGD 10 network (new SimpleProvider(features_data)) (new SimpleProvider(ys)) indexes 0.007 (Some((valfunc cvx target_class))))



  //elman_SGD_rmsprop  10 network  (new SimpleProvider(features_data))  (new SimpleProvider(ys)) indexes
  network

 (* let neuron_layer = new softmax_layer ((input_size),((output_size)))
  rprop_softmax  100 neuron_layer features_data ys

  neuron_layer*)


 member this.getModel () =
   (train_model)

 member this.save_model () =
  let model = this.getModel()
  model.Save(target_class+"_end.mod")
  File.WriteAllLines(target_class+".cls",features.Classes)


 

type nn_elman_predictor (features:features_provider,target_class:string,network:ITrainableNetwork)=

 let feat =
   let classes = File.ReadAllLines(target_class+".cls")
   features.Classes <- classes

   if features.FeaturesType <> "continous" then
    features.load_features_list ()

 let mutable model =

  let input_n =  features.Count()
  let output_n =  features.Classes.Length
  //Console.WriteLine("output size")
  //Console.WriteLine(features.Classes.Length)
  //Console.WriteLine("input size")
  //Console.WriteLine(input_n)

  //let network = new RCNN(input_n,input_n,output_n








  //let proj_layer = nm.AddProjectionLayer((wordVectors.base_dictionary.Count)+1,3,50,1,true)
  //nm.AddRecurrentLayer(20,53,2,false)
  //nm.AddSoftMaxLayer(output_n,20,(-1),false)
  //nm.FinalizeNet()


  //nm.AddTanhLayer(150,input_n,1,true)
  //nm.AddRecurrentLayer(10,6,2,false)
  //nm.AddTanhLayer(50,50,2,false,activation_relU)
  //nm.AddRecurrentLayer(50,input_n,1,true)
 // nm.AddRecurrentLayer(50,50,2,false,activation_relU )
  //nm.AddRecurrentLayer(70,70,2,true)
 // nm.AddProjectionLayer((wordVectors.base_dictionary.Count)+1,3,50,1,true)
 // nm.AddRecurrentLayer(20,53,1,false)
 // nm.AddSoftMaxLayer(output_n,20,(-1),false)
 // nm.FinalizeNet()

  //let network = new BRNN_Elman(25,input_n,output_n,false)
  //let network = new RCNN(input_n,input_n,output_n)

  Console.WriteLine( target_class + ".mod")
  network.Load (target_class + ".mod")

  network

 member this.LoadModel(ms:string) =
   model.Load(ms)



 member this.getError() =
  let indexes = features.Indexes()
  let inputs = features.compute_all_features_digital()
  let outputs = features.all_digital_output()
  model.BatchError (new SimpleProvider(inputs)) (new SimpleProvider(outputs)) indexes

 member this.predict_all_serial_universal() =
  let indexes = features.Indexes()
  let inputs = features.compute_all_features_digital()
  let mutable i = 0
  while i<(features.Entries.Count)  do
   let over, start = model.ForwardPass (new SimpleProvider(inputs)) indexes i
   for k = i to over - 1 do
    (features.Entries.[k]).[target_class+"p"] <- features.digital2class (model.OutputBuffer (k-i))

   i <- over


 member this.predict_integrate (modelid) =
  this.predict_all_serial_universal()
  let mutable buffer =""
  let mutable last_tag=""
  let entries = new structured_enitity_provider()
  let mutable current_entry = new Dictionary<string,string>()
  let data =  features.Entries
  for element in data do
   let word = element.["original"]

   let tag = element.[modelid + "p"]

  // System.Console.WriteLine(tag + " " + last_tag)
   if (tag = last_tag) && (tag<>"прочее" && tag<>"OTHER") then
    buffer <- buffer + word + " "

   if (tag <> last_tag) && (tag<>"прочее" && tag<>"OTHER") then
     if (last_tag<>"OTHER" && last_tag<>"") then
      current_entry.[last_tag] <- buffer
     // Console.WriteLine(buffer)
      entries.AddEntry(current_entry)

     current_entry <- new  Dictionary<string,string>()
     buffer <-  word + " "



   if (tag.Contains("прочее") || tag="OTHER") then
    if buffer<>"" && last_tag<>"OTHER" then
    //   Console.WriteLine(buffer + " ff")
       current_entry.[last_tag] <- buffer
       entries.AddEntry(current_entry)
       current_entry <- new  Dictionary<string,string>()
       buffer <- ""

   if word.Contains("stop") then
     entries.AddEntry(current_entry)
     current_entry <- new  Dictionary<string,string>()
     buffer <- ""

   last_tag <- tag
  entries.AddEntry(current_entry)
  entries


 (*member this.predict_all_serial_dyn() =
  for i=0 to (Array.length features.Entries) - 1 do
   let predict = this.predictmodel_dyn (features.compute_features_uni(i))
   (features.Entries.[i]).[target_class+"p"] <- predict *)



type accuracy_evalutor(predicted:structured_enitity_provider,reference:structured_enitity_provider,target_class) =
 
 member this.accuracy () =
  let a = Array.map2 (fun (x:Dictionary<string,string>) (y:Dictionary<string,string>) -> if x.[target_class] <> "BLANK" then (if x.[target_class] = y.[target_class] then 1.0 else 0.0) else -1.0) predicted.Entries reference.Entries
  let a1 = a |> Array.filter (fun x -> x > - 0.5)
  (Array.sum a1) / ((float)(Array.length a1))

 member this.showStats() =
  Console.WriteLine ("Overall accuracy = " + (this.accuracy()).ToString())

type accuracy_evalutor_p(predicted:structured_enitity_provider,target_class) =
 
 member this.accuracy () =
  let a = Array.map (fun (x:Dictionary<string,string>) ->  if x.[target_class] <> "BLANK" then (if x.[target_class] = x.[target_class+"p"] then 1.0 else 0.0) else -1.0) predicted.Entries
  let a1 = a |> Array.filter (fun x -> x > - 0.5)
  (Array.sum a1) / ((float)(Array.length a1))

 member this.recall typeclass =
  let a = Array.map (fun (x:Dictionary<string,string>) -> if (x.[target_class] = x.[target_class+"p"]) && (x.[target_class] = typeclass) then 1.0 else 0.0) predicted.Entries  
  let b = Array.map (fun (x:Dictionary<string,string>) -> if (x.[target_class] = typeclass) then 1.0 else 0.0) predicted.Entries  
  (Array.sum a) / (Array.sum b)
 
 member this.prec typeclass =
  let a = Array.map (fun (x:Dictionary<string,string>) -> if (x.[target_class] = x.[target_class+"p"]) && (x.[target_class] = typeclass) then 1.0 else 0.0) predicted.Entries  
  let b = Array.map (fun (x:Dictionary<string,string>) -> if (x.[target_class+"p"] = typeclass) then 1.0 else 0.0) predicted.Entries  
  (Array.sum a) / (Array.sum b)
 
 member this.countclass typeclass =
  let a = Array.map (fun (x:Dictionary<string,string>) -> if (x.[target_class+"p"] = typeclass) then 1.0 else 0.0) predicted.Entries  
  (Array.sum a)
  
 member this.f1 typeclass =
  2.0 * ((this.prec typeclass) * (this.recall typeclass)) / ((this.prec typeclass) + (this.recall typeclass))
  
 member this.showStats tp =
  Console.WriteLine ("Overall accuracy = " + (this.accuracy()).ToString())
  Console.WriteLine ("Recall for class = " + (this.recall tp).ToString())
  Console.WriteLine ("Precession for class = " + (this.prec tp).ToString())
  Console.WriteLine ("F1 score for class = " + (this.f1 tp).ToString())

let mutable prev_acc = 0.0
let mutable prev_error = 1000000.0

let cv_validate_elman (dataset:features_provider) (target_class:string) (network:ITrainableNetwork)   =
 network.Save("network.tmp")
 let predictor = new nn_elman_predictor(dataset,target_class,network)

 predictor.LoadModel("network.tmp")
 predictor.predict_all_serial_universal()
 let evalute_cv = new accuracy_evalutor_p(dataset.DataSet,target_class)
 let acc = evalute_cv.accuracy()
 let error = predictor.getError()
 Console.Write(" |CV err=" + (error.ToString()) + " + acc=" + (if acc.ToString().Length > 4 then ((acc.ToString()).[0..4]) else (acc.ToString())))
 //network.Save(mydir + target_class + ".mod")
 if acc>prev_acc then
  prev_acc <- acc
  network.Save(target_class + ".mod")
  Console.Write(" yes")
 else
  Console.Write(" no")

 if error<prev_error then
  Console.Write(" yes")

  prev_error <- error
 else
  Console.Write(" no")

 Console.WriteLine()
