module wordVectors
open System
open System.IO
open System.Collections.Generic
open System.Text.RegularExpressions




let fp_is_number line =
   let pp = Regex.Matches(line,"[\\d]{3,6}(\.[\d]0){0,1}")
   (pp.Count=1) && (pp.[0].Index=0) && (pp.[0].Length = line.Length)

//word-shapes
//word shapes add-on
//features generator
let readfeature_dict filename =
 let p = File.ReadAllLines(filename)
 let dict = new Dictionary<string,string>()
 for px in p do
  dict.[px] <- px
 dict

let latUp =  readfeature_dict "latinUP.txt"
let latDown =  readfeature_dict "latinDown.txt"
let ruUp =  readfeature_dict "russianUP.txt"
let ruDown =  readfeature_dict "russianDown.txt"
let digits =   readfeature_dict "numbers.txt"

//word-shapes

let check_symbol (symb:string) =
   let mutable symtype="o"
   if latUp.ContainsKey(symb) then symtype <- "L"

   if latDown.ContainsKey(symb) then symtype <- "l"

   if ruUp.ContainsKey(symb) then symtype <- "R"

   if ruDown.ContainsKey(symb) then symtype <- "r"

   if digits.ContainsKey(symb) then symtype <- "D"

   symtype

let genWordShape (line:string) =
 let mutable wshape = ""
 let mutable psym = ""
 let mutable pindex = 0
 let mutable i = 0
 for i = 0 to line.Length - 1 do
   let sym = (line.[i]).ToString()
   let symtyp = check_symbol sym
   if psym <> symtyp then
    wshape <- wshape + symtyp
   psym <- symtyp
 "##" + wshape


// end shapes add-on


let vector_add (v1:float array) (v2:float array)  =
  Array.map2 (fun x y -> x+y)  v1 v2

let vector_sub (v1:float array) (v2:float array)  =
  Array.map2 (fun x y -> x-y)  v1 v2

let normalize_vector (vect:float array) =
 let mag = vect |> Array.map (fun x -> x*x) |> Array.sum |> Math.Sqrt
 vect |> Array.map (fun x -> x/mag)

let projections = File.ReadAllLines ("features.vect") |> Array.map (fun x -> x.Split([|' '|],StringSplitOptions.RemoveEmptyEntries) |> Array.map (fun x -> (float) (x.Replace(",",".")))) |> Array.map (fun y -> Array.map (fun x -> Math.Tanh x) y) //|> Array.map normalize_vector //Array.map (fun y -> Array.map (fun x -> Math.Tanh x) y) |>


let dot_product (a:array<'a>) (b:array<'a>) =
    if Array.length a <> Array.length b then failwith "invalid argument: vectors must have the same lengths"
    Array.fold2 (fun acc i j -> acc + (i * j)) 0.0 a b



let normalize (a:float array) =
  let sum = Math.Sqrt (a |> Array.map (fun x -> x*x) |> Array.sum )
  a |> Array.map (fun x -> x / sum)

let cosine_similarity  (vect:float array)  (vect1:float array) =
   dot_product  (normalize vect)  (normalize vect1)



let base_dictionary =
  let p = (File.ReadAllLines("features.dict"))
  let dict = new Dictionary<string,int>()
  for i = 0 to p.Length - 1 do
   dict.[p.[i].ToLower()] <- i
  dict

let has_projection (word1:string) =
  let word = word1.ToLower()
  base_dictionary.ContainsKey(word)

let get_word_index (word1:string) =
 let word = word1.ToLower()

 if base_dictionary.ContainsKey(word) then
    base_dictionary.[word]
  else
    base_dictionary.["c'est"]

let project_word (word1:string) =
 // let word = stem_word (word1.ToLower())
  let word =   (word1.ToLower())

  if base_dictionary.ContainsKey(word) then
    projections.[base_dictionary.[word]]
  else
    projections.[base_dictionary.["c'est"]]


let add_vectors (vect1:float array) (vect2:float array) =
  for i=0 to vect2.Length - 1 do
   vect1.[i] <- vect1.[i]  + vect2.[i]

let distance (vect:float array)  (vect1:float array) =
 Array.map2 (fun x y -> (x-y)*(x-y)) vect vect1 |> Array.sum |> Math.Sqrt

let project_text (text:string) =
 // let split = Regex.Split((text.Replace("\n"," ")), @"(?=[–/\:\.\;\[\]\,\-\(\)\s\!\?\!\""\\])|(?<=[–/\:\.\;\[\]\,\-\(\)\s\!\?\!\""\\])") |> Array.map (fun x -> ((x.ToLower()).Trim())) |> Array.filter (fun x -> x.Length > 0)
  let split = text.Split([|" ";",";".";";";")";"(";"-"|],StringSplitOptions.RemoveEmptyEntries)
  let vector = Array.init 50 (fun x -> 0.0)
  let mutable cnt = 0.0
  for wrd in split do
   //if base_dictionary.ContainsKey(wrd) then
    let vect = project_word wrd
    add_vectors vector vect
    cnt <- cnt + 1.0
  let cnt1 = cnt
  let count = float (split.Length)
  vector |> Array.map (fun x -> x / count)  |> normalize_vector  |> (Array.append ([|count/25.0|]))
 // vector |> normalize_vector


let project_text_mod (text:string) =
 // let split = Regex.Split((text.Replace("\n"," ")), @"(?=[–/\:\.\;\[\]\,\-\(\)\s\!\?\!\""\\])|(?<=[–/\:\.\;\[\]\,\-\(\)\s\!\?\!\""\\])") |> Array.map (fun x -> ((x.ToLower()).Trim())) |> Array.filter (fun x -> x.Length > 0)
  let split = text.Split([|" ";",";".";";";")";"(";"-"|],StringSplitOptions.RemoveEmptyEntries)
  let vector = Array.init 50 (fun x -> 0.0)
  let mutable cnt = 0.0
  for wrd in split do
   if base_dictionary.ContainsKey(wrd) then
    let vect = project_word wrd
    add_vectors vector vect
    cnt <- cnt + 1.0
  let cnt1 = cnt
  let count = float (split.Length)
 // vector |> Array.map (fun x -> x / count)  |> normalize_vector  |> (Array.append ([|count/25.0|]))
  vector |> normalize_vector


let vector2string (p: float array) =
   p |> Array.map (fun x -> (x.ToString()) + " ") |> Array.fold (+) ""

let string2vector (p:string) =
 //Console.WriteLine (p)
 p.Split([|" "|],StringSplitOptions.RemoveEmptyEntries) |> Array.map (fun x -> (float)(x.Replace(",",".")))
