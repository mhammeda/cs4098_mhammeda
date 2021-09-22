module Lib
    ( someFunc
    ) where
import System.Environment
import Data.Set (Set, fromList, toList, insert, empty)
import Control.Monad.State
import Data.List (intersperse, (\\))

someFunc :: IO ()
someFunc = do
  [f] <- getArgs
  fc <- readFile f
  let adjlist = parseAdjacencyList fc
  putStrLn $ makeGraph adjlist

parseAdjacencyList :: String -> AdjacencyList
parseAdjacencyList f =
  let ls = lines f
      as = filter (\l -> not (head l == '#')) ls
      ws = words <$> as
      is = (read <$>) <$> ws
    in AdjacencyList is

newtype AdjacencyList = AdjacencyList [[Int]]
  deriving Show
  
newtype EdgeSet = EdgeSet (Set (Set Int))
  deriving Show

graphSize :: AdjacencyList -> Int
graphSize (AdjacencyList l) = 1 + maximum (join l)

toEdgeSet :: AdjacencyList -> State EdgeSet ()
toEdgeSet (AdjacencyList []) = return () 
toEdgeSet (AdjacencyList (a:as)) = toEdgeSet' a >> toEdgeSet (AdjacencyList as)
  where
    toEdgeSet' :: [Int] -> State EdgeSet ()
    toEdgeSet' [] = error "toEdgeSet': empty adjacency list!"
    toEdgeSet' [s] = return ()
    toEdgeSet' (f:r) = mapM_ addEdge $ zip (cycle [f]) r
    addEdge :: (Int,Int) -> State EdgeSet ()
    addEdge (l,r) = do
      EdgeSet s <- get
      put $ EdgeSet $ insert (fromList [l,r]) s

makeGraph :: AdjacencyList -> String
makeGraph a = "{\"nodes\" : " ++ makeNodes a ++ ",\"links\" :" ++ makeLinks a ++ "}"

makeNodes :: AdjacencyList -> String
makeNodes a = "[" ++ join (intersperse "," (nodeObj <$> [0..graphSize a])) ++ "]"
  where
    nodeObj :: Int -> String
    nodeObj i = "{\"id\" : " ++ show i ++ "}"

makeLinks :: AdjacencyList -> String
makeLinks a = 
   "[" ++ join (intersperse "," [ toEdge e | e <- toList es]) ++ "]"
  where (EdgeSet es) = execState (toEdgeSet a) (EdgeSet empty) 
        toEdge :: Set Int -> String
        toEdge s = case toList s of
                     [l,r] -> "{\"source\" : " ++ show l
                           ++ "," ++ "\"target\" : " ++ show r ++ "}"
                     e -> error $ "makeLinks: toEdge: wrong dimension: " ++ show e
 

