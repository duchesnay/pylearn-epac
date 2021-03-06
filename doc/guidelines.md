Architecture
============

- Découpler les composants entre eux ainsi que l'utilisation de composants externes (ex: soma-workflow, pyanatomist, vip etc...). Cela peut se faire par l’ajout d’une interface entre l’interface du composant (externe ou non) et du reste de l’application.
- Dans le cas d’un composant externe, il faut prendre la précausion de concevoir cette interface “du point de vue” de l’application. Cela signifie qu’elle reflètera les besoins de l’application par rapport à ce composant.
- Pour aider au découplage il est bon de s’imaginer que chaque composant peut avoir différentes implémentations et que chaque dépendance externe peut etre remplacée.


Guideline générale
==================

- Fonction plutot petites que grandes (à partir de 25 lignes il est bon de se poser des questions)
- Favoriser des noms discriminants pour la recherche dans une fonction, une classe, dans un fichier ou un ensemble de fichiers (le debug et les renomages sont facilités)
- Commentaires limités (mieux vaut clarifier le code ou renommer de facon plus explicite que d’ajouter un commentaire)
- Certains commentaires sont importants comme l’explication de certains choix.
- Utiliser les commentaires standards (et donc faciles à rechercher) quand ils sont adéquats : TODO, FIXME, XXX
- Boy scout rule: Leave the campground cleaner than you found it. 
- Lecture des méthodes de haut en bas. Ex: si la méthode A utilise la méthode B, écrire A puis B.
- Ordre des classes : top -> down, code client -> code interne, code haut niveau -> code bas niveau.
- Ne pas laisser de prints (stdout propre). 

Format de code Python
=====================

- Pep 8 
- 80 caractères pas strict mais recommandé, et strictement inférieur à 120 caractères.
- un module fait au maximum 400 lignes
- Tous les membres d’une instance de classe doivent être déclarés explicitement dans la fonction __init__ ou dans des méthodes préfixées par “_init_” et appelées par le constructeur  (pas d’ajout à la volée).
- Pas d’”import *”
- Pas d’import relatif
- Déclarer les imports standards (eg. python) en premier, puis après une ligne blanche les imports externes (eg. aims, soma_workflow) et enfin après une ligne blanche les imports internes.
- Utiliser un nommage cohérent avec les autres module pour les “import something as smthg”
- Les methodes et membres considérés comme private sont préfixés par deux underscores (règle non utilisée pour l’instant). Les methodes et membres considérés comme protected sont préfixés par un underscore. Pour l’accès par réference const (modification de l’objet autorisé mais pas de la référence), utiliser les properties.
- Les méthodes “friends” sont préfixées par «_friend» afin d’indiquer qu’elles sont protégées mais accessible pour des besoins d’architecture à d’autres classes, mais pas aux utilisateurs.
- Les listes sont indiquées par un nom au pluriel
- Pour distinguer les nom de fichiers avec le path ou pas, absolu ou relatif, utiliser la convention suivante :
    * filename : le nom du fichier sans le chemin (basename)
    * filepath : le nom du fichier avec le chemin absolu
    * file_relative_path : le nom du fichier avec un chemin relatif 


Commit Message
==============

Formatage git
-------------

- une ligne synthétique commençant obligatoirement par un des tags ci-dessous
- une ligne blanche (sera ignorée par git)
- un commentaire plus détaillé sur plusieurs lignes


Liste des tags pour la première ligne
-------------------------------------

- **ENH**: When adding or improving an existing or new class in term of capabilities,
- **COMP**: When fixing a compilation error or warning,
- **DOC**: When starting or improving the class documentation,
- **STYLE**: When enhancing the comments or coding style without any effect on the class behaviour,
- **REFAC**: When refactoring without adding new capabilities,
- **BUG**: When fixing a bug (if the bug is identified in the tracker, please add the reference),
- **INST**: When fixing issues related to installation,
- **PERF**: When improving performance,
- **TEST**: When adding or modifying a test,
- **WRG**: When correcting a warning.




