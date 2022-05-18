# TIPE

TIPE année 2022/2023
Thème : Optimisation de trajectoire

L'objectif du TIPE consiste en l'optimisation du trajet d'un point A à un point B,suivant diverses problématiques : temps de calcul de l'algorithme, distance réelle du trajet, réalisme de l'algorithme vis à vis d'une situation donnée, etc...
En premier lieu, on s'intéresse à un environnement polygonal simple (partie "TIPE - Polygone"). Dans un tel environnement, il s'avère alors pertinent de s'intéresser à la triangulation du polygone pour déterminer un trajet avec un nombre de calculs moindre. En effet, la triangulation s'effectue en 0(n²) (voire en O(nlog(n)) pour des algorithmes encore plus sophistiqués) et on détermine alors le chemin résultant avec un parcours en largeur des triangles adjacents ; algorithme de complexité 0(n²). Donc une complexité finale en O(n²) !
Ensuite, il est abordée la recherche optimale
