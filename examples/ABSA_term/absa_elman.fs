module ABSA
open System
open System.IO
open System.Collections.Generic
open System.Text.RegularExpressions
open DataAccess
open FeaturesProvider
open  wordVectors


open NeuThink.Neuron
open NeuThink.NeuronTraining
open NeuThink.NeuralLayers
open NeuThink.DataSources




let GenerateRNN input_size output_size =
 let network = new RecurrentNetwork()

 network.AddPerceptronLayer(60,input_size,[|(1)|],true,0)
 network.AddSoftMaxLayer(output_size,60,[|(-1)|],false,1)

 network.FinalizeNet()
 network

let last2symbols (x:string) =
  let r = (x.Length - 1)
  if r>2 then
   (x.[r-1].ToString()) + (x.[r].ToString())
  else
   ""

let last3symbols (x:string) =
  let r = (x.Length - 1)
  if r>4 then
   (x.[r-2].ToString()) + (x.[r-1].ToString()) + (x.[r].ToString())
  else
   ""

let rf_is_number line =
   let pp = Regex.Matches(line,"[№N]{0,1}[\\d]{1,6}(\.[\d]0){0,1}[сC]{0,1}")
   (pp.Count=1) && (pp.[0].Index=0) && (pp.[0].Length = line.Length)

let rf_is_small_number line =
   let pp = Regex.Matches(line,"[\\d]{4}(\.[\d]0){0,1}")
   (pp.Count=1) && (pp.[0].Index=0) && (pp.[0].Length = line.Length)

let rf_year line =
   let pp = Regex.Matches(line,"(19[\\d]{2})||(20[0-2][\\d])[г]{0,1}")
   (pp.Count=1) && (pp.[0].Index=0) && (pp.[0].Length = line.Length)

let rf_latin_symbols line =
   let pp = Regex.Matches(line,"([A-Z][a-z]{2,11}([A-Z][a-z]{2,4}){0,1})|([A-Z]{5,9}|([a-z][\-][A-Z]{5,6})|HP|OKI|Pixma)")
   (pp.Count>0) && (pp.[0].Index=0) && (pp.[0].Length = line.Length)

let start (l1:string) =

 if l1.Contains("<start>") then
  true
 else
  false

let keys = File.ReadAllLines("keywords.txt")

let positives = File.ReadAllLines("lexicon_pos.txt") |> Array.map (fun x -> x.Trim())
let negatives = File.ReadAllLines("lexicon_neg.txt") |> Array.map (fun x -> x.Trim())

let check_pr (word:string) (lex:string array) (pword:string) =
 let mutable k =  -1.0
 for i = 0 to lex.Length - 1  do
  if (word.ToLower()).Contains(lex.[i]) then
   k <- 1.0
 if pword="не" then k <- -k
 k

let gen_vector_keys(sent:string) =
 let ln = (float ((sent.Split([|" "|],StringSplitOptions.RemoveEmptyEntries)).Length)) / 25.0
 Array.append (keys |> Array.map (fun k -> if (sent.ToLower()).Contains(k) then 1.0 else 0.0))  [|ln|]


let genFeatures_cont (data:Dictionary<string,string> array) (i:int) =
 let features = new ResizeArray<string>()
 let BaseWord = (data.[i]).["base_word"]
 let ConnectedWord =  (data.[i]).["connected_word"]
 let direct = if ((int)(data.[i].["distance"])) < 0 then -1.0 else 1.0
 let length =  (6.0/(float)(data.[i].["distance"]))-2.0
 Array.append (Array.append (project_word  BaseWord)  (project_word ConnectedWord)) [|direct;length|]


let genFeatures_cont_serial ((data:ResizeArray<Dictionary<string,string>>)) (i:int) =
 let features = new ResizeArray<string>()
 let CurrentWord = (data.[i]).["word"]
 let future_word = if data.Count > i+1 then (data.[i+1]).["word"] else "unk"
 let future2_word = if data.Count > i+2 then (data.[i+2]).["word"] else "unk"

 let prev_word  = if  i>0 then (data.[i-1]).["word"] else "unk"
 let ar2 = [|(check_pr CurrentWord positives prev_word);(check_pr CurrentWord negatives prev_word)|]
 let bias = if data.[i].["target"] = "OTHER" then [|1.0|] else [|-1.0|]
 (project_word  CurrentWord)
 //Array.append (Array.append (project_word  CurrentWord) (project_word  future_word)) (project_word  prev_word)) //bias
 //(Array.append (project_word  CurrentWord) (project_word  prev_word))

 //(Array.append (project_word  CurrentWord) (project_word  future_word))


let genFeatures_cont_serial_proj ((data:ResizeArray<Dictionary<string,string>>)) (i:int) =
 let CurrentWord = (data.[i]).["word"]
 let index =  get_word_index CurrentWord
 [|(float index)|]



let rnd = new Random(65536)




let textlets1 = new ResizeArray<ResizeArray<string>>()

let get_textlets() =
 let multi_prov = new structured_enitity_provider()
 multi_prov.load_multi_output("spolar_data.txt")
 multi_prov

let getdistance word ix (textlet:ResizeArray<string>) =
 let mutable dist = 0
 for i=0 to textlet.Count - 1 do
   let word1 = textlet.[i]
   if word = word1 then
    dist <- ix - i
   dist


let abs x = if x < 0 then -x else x

let textlet2dict_simple (textlet:Dictionary<string,string>) =
 let dicts = new ResizeArray<Dictionary<string,string>>()
 let text = textlet.["text"]
 //let str_vect = vector2string (project_text text)
 let str_vect = vector2string (gen_vector_keys text)
 let num_vect =  gen_vector_keys text
 let ps = Regex.Split(text, @"(?=[–/\:\.\;\[\]\,\-\(\)\s\\])|(?<=[–/\:\.\;\[\]\,\-\(\)\s\\])") |> Array.map (fun x -> (x.Trim()).ToLower()) |> Array.filter (fun x -> x.Length > 0)
 let mutable counter = 0.0
 let leng = float (ps.Length - 1) / 25.0
 let st_dict = new Dictionary<string,string>()
 let mutable pstr = ""
 if ps.Length > 0 then
  st_dict.["word"] <- "<START>"
  st_dict.["target"] <- ps.[0]
  st_dict.["descriptor"] <- str_vect
  st_dict.["counter"] <- counter.ToString()
  st_dict.["length"] <- (leng.ToString())
  st_dict.["index"] <- "0"
  dicts.Add(st_dict)
  for i=0  to ps.Length - 1 do
   counter <- counter + (1.0 / 25.0)
   let dict = new Dictionary<string,string>()

   dict.["descriptor"] <- vector2string (vector_sub num_vect  (gen_vector_keys pstr) )
  // dict.["prod_vector"] <- vector2string (gen_vector_keys pstr)
   dict.["counter"] <- counter.ToString()
   dict.["length"] <- (leng.ToString())
   dict.["index"] <- ((i + 1).ToString())
   if (i+1) < ps.Length - 1 then
    dict.["word"] <- ps.[i]
    dict.["target"] <- ps.[i+1]
   else
    dict.["word"] <- ps.[i]
    dict.["target"] <- "<END>"
   pstr <- pstr + " " + ps.[i]

   if dict.ContainsKey("word") then
    dicts.Add(dict)
 dicts

let gen_data_all (source:structured_enitity_provider) =
 let provider_test = new structured_enitity_provider()
 let provider_train = new structured_enitity_provider()
 let mutable i = 0
 for t in source.Entries do
  let data =  textlet2dict_simple t
  let p = rnd.Next(10)
  if i < 1100 then
   if p>6 then
    for d in data do
     provider_test.AddEntry(d)
   else
    for d in data do
     provider_train.AddEntry(d)
  i <- i + 1
 (provider_test,provider_train)


let sequenazier (x:structured_enitity_provider) =
 let mutable index = 0
 for i = 0 to x.Entries.Length - 1 do
  let entry = x.getEntry i
  let prev_word = if i>0 then (x.getEntry (i-1)).["word"] else ""
  if entry.["word"] = "<STOP>" && (prev_word <> "<STOP>") then
   index <- 0
  entry.["index"] <- index.ToString()
  index <- index + 1


let () =
 //let data = get_textlets()
 let data_test = new structured_enitity_provider()
 let data_train = new structured_enitity_provider()

 data_train.load_serial_input "ABSA-15_Restaurants_razmetka.txt"
// data_test.load_serial_input_unf "cv_stemmed_expl.txt"
 data_test.load_serial_input "Restaurants_Test_Gold_col.txt"
// st_arm_marking_for_all_0_stemmed.txt
 sequenazier  data_train
 sequenazier  data_test
 Console.WriteLine("Saving data")
 data_test.save_serial_output "test.txt"

 //Train model
 Console.WriteLine("Training model")
 let network = GenerateRNN 80 2
 
 let ftrain =  new features_provider(data_train,"target")
 let ft_test = new features_provider(data_test,"target")
 ft_test.FeaturesType <- "continous"
 ftrain.FeaturesType <- "continous"
 //Somewhere here change features function

 ftrain.generator_continuos <- genFeatures_cont_serial
 ft_test.generator_continuos <- genFeatures_cont_serial

 //Model Generator
 let trainer = new elman_trainer(ftrain,"target",network,Some(ft_test,cv_validate_elman))
 //trainer.save_model()
 //obtain prediction
 Console.WriteLine("Predicting")

 let predictor_train = new nn_elman_predictor(ftrain,"target",network)
 let predictor_test = new nn_elman_predictor(ft_test,"target",network)
 predictor_train.predict_all_serial_universal()
 predictor_test.predict_all_serial_universal()
 //Evalute accuracy
 let evalute_train = new accuracy_evalutor_p(data_train,"target")
 Console.WriteLine("TRAIN,EXPLICIT")
 evalute_train.showStats("explicit")
 Console.WriteLine("TRAIN,implicit")
 evalute_train.showStats("implicit")

 Console.WriteLine("CV")
 let evalute_test = new accuracy_evalutor_p(data_test,"target")
 evalute_test.showStats("explicit")
 Console.WriteLine("CV,IMPLICIT")
 evalute_test.showStats("implicit")
 Console.WriteLine("CV,FACTUAL")
 evalute_test.showStats("fct")
 Console.WriteLine("CV,OTHER")
 evalute_test.showStats("OTHER")

 (*Console.WriteLine("TRAIN,REL")
 evalute_train.showStats("Rel")
 //Console.WriteLine("TRAIN,NEG")
 //evalute_train.showStats("negative")

 Console.WriteLine("CV")
 let evalute_test = new accuracy_evalutor_p(data_test,"target")
 evalute_test.showStats("Rel") *)


 //show_time_profile ()

 //Save result for examination
 data_test.save_serial_output "test.txt"
 data_train.save_serial_output "test_train.txt"

