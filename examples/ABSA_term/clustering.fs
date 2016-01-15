open System
open System.IO
open  wordVectors
open GenSentence


let data = File.ReadAllLines("positives.txt") |> Array.map (fun x -> ((project_text_mod x),x))
let indexes = new ResizeArray<int>()

let check_index i =
 Array.exists (fun x -> x=i) (Array.ofSeq indexes)

let cluster_data () =
 let clusters = new ResizeArray<ResizeArray<float array*string>>()
 for i=0 to data.Length - 1 do
  if (not (check_index i)) then
   let nc = new ResizeArray<float array*string>()
   nc.Add(data.[i])
   indexes.Add(i)
   for j= i+1 to data.Length - 1 do
    let sim = cosine_similarity (fst (data.[i])) (fst (data.[j]))
    if (sim>0.79) && not (check_index j) then
     nc.Add(data.[j])
     indexes.Add(j)
   clusters.Add(nc)
 clusters

let project_cluster (clust:ResizeArray<float array*string>)  =
 let vect = Array.create 51 0.0
 let vectors = (Array.ofSeq clust) |> Array.map (fun x -> project_text (snd x))
 for cl in vectors do
  add_vectors vect cl
 let count = (float)(clust.Count)
 vect |> Array.map (fun x -> x / count)  |> normalize_vector

let () =
 let cl = cluster_data ()
 for clust in cl do
  Console.WriteLine("--------------")
  //for dta in clust do
  // Console.WriteLine ((snd dta))
  if clust.Count>1 then
   let dvect =  project_cluster clust
   ignore(generate_sentence (vector2string dvect))
   Console.WriteLine(">>11111111>>")
   for dta in clust do
     Console.WriteLine ((snd dta))




