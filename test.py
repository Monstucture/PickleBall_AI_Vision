from collections import deque


grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]
]

count = 0 
rows = len(grid)
cols = len(grid[0])
visited = set()

if not grid:
	print(0)

def bfs(r,c):
	q = deque()
	visited.add((r,c))
	q.append((r,c))
	
	while q:
		row, col = q.popleft()	
		directions = [(0,1), (1,0), (0,-1), (-1,0)]
		
		for dr, dc in directions:
			r, c = row + dr, col + dc
			if r in range(rows) and c in range(cols) and grid[r][c] == "1" and (r,c) not in visited:
				q.append((r,c))
				visited.add((r,c))


for r in range(rows):
	for c in range(cols):
		if grid[r][c] == "1" and (r,c) not in visited:
			bfs(r,c)
			count+=1

print(count)