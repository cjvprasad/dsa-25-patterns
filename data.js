// --- DATA STRUCTURES ---
      const phases = [
        { title: "Phase 1: The Foundation", id: "p1" },
        { title: "Phase 2: Core Structures", id: "p2" },
        { title: "Phase 3: Advanced Search", id: "p3" },
        { title: "Phase 4: Dynamic Programming", id: "p4" },
        { title: "Phase 5: Expert Graph Theory", id: "p5" },
      ];

      // UPDATED PATTERNS (With formatted Anchor Logic)
      const patterns = [
        {
          id: 1,
          phase: "p1",
          name: "Sliding Window",
          desc: "Used to perform a required operation on a specific window size of a given array or linked list.",
          strategy:
            "Use 'start' and 'end' pointers. Expand 'end' until condition breaks, then shrink 'start'.",
          anchor: {
            logic:
              "let windowSum = 0, start = 0;\nfor (let end = 0; end < arr.length; end++) {\n  windowSum += arr[end];\n  while (windowSum > target) {\n    windowSum -= arr[start];\n    start++;\n  }\n}",
          },
          problems: [
            "Maximum Sum Subarray of Size K",
            "Smallest Subarray with a given sum",
            "Longest Substring with K Distinct Characters",
            "Fruits into Baskets",
          ],
        },
        {
          id: 2,
          phase: "p1",
          name: "Two Pointers",
          desc: "Used in sorted arrays to find a set of elements that fulfill a certain constraint.",
          strategy:
            "Initialize two pointers (Left=0, Right=N-1). Move them based on sum comparison.",
          anchor: {
            logic:
              "let left = 0, right = arr.length - 1;\nwhile (left < right) {\n  const sum = arr[left] + arr[right];\n  if (sum === target) return [left, right];\n  else if (sum < target) left++;\n  else right--;\n}",
          },
          problems: [
            "Pair with Target Sum",
            "Remove Duplicates from Sorted Array",
            "Squaring a Sorted Array",
            "3Sum",
          ],
        },
        {
          id: 3,
          phase: "p1",
          name: "Fast & Slow Pointers",
          desc: "Also known as Hare & Tortoise. Used for cycle detection in linked lists.",
          strategy:
            "Move 'fast' pointer 2 steps and 'slow' pointer 1 step. If they meet, there is a cycle.",
          anchor: {
            logic:
              "let slow = head, fast = head;\nwhile (fast && fast.next) {\n  slow = slow.next;\n  fast = fast.next.next;\n  if (slow === fast) return true;\n}\nreturn false;",
          },
          problems: [
            "Linked List Cycle",
            "Middle of the Linked List",
            "Start of Linked List Cycle",
            "Happy Number",
          ],
        },
        {
          id: 4,
          phase: "p1",
          name: "Merge Intervals",
          desc: "Used to deal with overlapping intervals. Efficient for scheduling.",
          strategy:
            "Sort intervals by start time. Iterate and merge if curr.start <= prev.end.",
          anchor: {
            logic:
              "intervals.sort((a, b) => a[0] - b[0]);\nconst merged = [intervals[0]];\nfor (let i = 1; i < intervals.length; i++) {\n  if (curr.start <= prev.end) {\n    prev.end = Math.max(prev.end, curr.end);\n  } else merged.push(curr);\n}",
          },
          problems: [
            "Merge Intervals",
            "Insert Interval",
            "Intervals Intersection",
            "Conflicting Appointments",
          ],
        },
        {
          id: 5,
          phase: "p1",
          name: "Cyclic Sort",
          desc: "Used when numbers are in a given range (1 to N) to find duplicates/missing.",
          strategy:
            "Iterate array. Swap nums[i] to correct index nums[nums[i]-1].",
          anchor: {
            logic:
              "let i = 0;\nwhile (i < n) {\n  const j = nums[i] - 1;\n  if (nums[i] !== nums[j]) swap(i, j);\n  else i++;\n}",
          },
          problems: [
            "Missing Number",
            "Find All Numbers Disappeared in an Array",
            "Find the Duplicate Number",
            "Find All Duplicates in an Array",
          ],
        },
        {
          id: 6,
          phase: "p2",
          name: "List Reversal",
          desc: "Reverse a Linked List in-place, completely or within a range.",
          strategy:
            "Use 'prev', 'curr', and 'next' pointers. Rotate them as you iterate.",
          anchor: {
            logic:
              "let prev = null, curr = head;\nwhile (curr) {\n  let next = curr.next;\n  curr.next = prev;\n  prev = curr;\n  curr = next;\n}\nreturn prev;",
          },
          problems: [
            "Reverse Linked List",
            "Reverse Linked List II",
            "Reverse Nodes in k-Group",
            "Rotate List",
          ],
        },
        {
          id: 7,
          phase: "p2",
          name: "Tree BFS",
          desc: "Breadth-First Search. Traverse a tree level-by-level.",
          strategy: "Use a Queue. Process 'levelSize' nodes at a time.",
          anchor: {
            logic:
              "const q = [root];\nwhile (q.length) {\n  const len = q.length;\n  for (let i = 0; i < len; i++) {\n    const node = q.shift();\n    if (node.left) q.push(node.left);\n    if (node.right) q.push(node.right);\n  }\n}",
          },
          problems: [
            "Binary Tree Level Order Traversal",
            "Binary Tree Zigzag Level Order Traversal",
            "Average of Levels in Binary Tree",
            "Minimum Depth of Binary Tree",
          ],
        },
        {
          id: 8,
          phase: "p2",
          name: "Tree DFS",
          desc: "Depth-First Search. Go deep before backtracking.",
          strategy:
            "Recursion. Process Node, then recursively call DFS on children.",
          anchor: {
            logic:
              "function dfs(node) {\n  if (!node) return;\n  // Process node\n  dfs(node.left);\n  dfs(node.right);\n}",
          },
          problems: [
            "Path Sum",
            "Path Sum II",
            "Path Sum III",
            "Sum Root to Leaf Numbers",
          ],
        },
        {
          id: 9,
          phase: "p2",
          name: "Two Heaps",
          desc: "Find median of a stream or min/max problems efficiently.",
          strategy: "MinHeap (larger half) and MaxHeap (smaller half).",
          anchor: {
            logic:
              "if (num <= maxH.peek()) maxH.push(num);\nelse minH.push(num);\n// Balance sizes\nif (maxH.size > minH.size + 1) minH.push(maxH.pop());",
          },
          problems: [
            "Find Median from Data Stream",
            "Sliding Window Median",
            "IPO",
            "Find Right Interval",
          ],
        },
        {
          id: 10,
          phase: "p2",
          name: "Subsets",
          desc: "Permutations and combinations using BFS.",
          strategy:
            "Start with empty set. For each number, add it to existing subsets.",
          anchor: {
            logic:
              "const subsets = [[]];\nfor (const n of nums) {\n  const len = subsets.length;\n  for (let i = 0; i < len; i++) {\n    subsets.push([...subsets[i], n]);\n  }\n}",
          },
          problems: [
            "Subsets",
            "Subsets II",
            "Permutations",
            "Letter Combinations of a Phone Number",
          ],
        },
        {
          id: 11,
          phase: "p3",
          name: "Binary Search",
          desc: "Find target in sorted array in O(log N).",
          strategy:
            "Calculate mid. If target < mid, search left. Else search right.",
          anchor: {
            logic:
              "let l = 0, r = arr.length - 1;\nwhile (l <= r) {\n  const mid = l + Math.floor((r - l) / 2);\n  if (arr[mid] === t) return mid;\n  if (arr[mid] < t) l = mid + 1;\n  else r = mid - 1;\n}",
          },
          problems: [
            "Binary Search",
            "Search Insert Position",
            "Search in Rotated Sorted Array",
            "Find Peak Element",
          ],
        },
        {
          id: 12,
          phase: "p3",
          name: "Bitwise XOR",
          desc: "Find unique numbers or missing numbers efficiently.",
          strategy: "XOR of a number with itself is 0. XOR with 0 is number.",
          anchor: {
            logic: "let res = 0;\nfor (const n of nums) res ^= n;\nreturn res;",
          },
          problems: [
            "Single Number",
            "Single Number II",
            "Hamming Distance",
            "Total Hamming Distance",
          ],
        },
        {
          id: 13,
          phase: "p3",
          name: "Top K Elements",
          desc: "Find top/smallest/frequent K elements.",
          strategy: "MinHeap (size K) for K largest. MaxHeap for K smallest.",
          anchor: {
            logic:
              "for (const n of nums) {\n  minHeap.push(n);\n  if (minHeap.size() > k) minHeap.pop();\n}",
          },
          problems: [
            "Kth Largest Element in an Array",
            "Top K Frequent Elements",
            "K Closest Points to Origin",
            "Sort Characters By Frequency",
          ],
        },
        {
          id: 14,
          phase: "p3",
          name: "K-way Merge",
          desc: "Merge K sorted lists.",
          strategy:
            "MinHeap with first element of each list. Pop min, push next.",
          anchor: {
            logic:
              "heap.push({val: list[0], idx: 0});\nwhile(heap) {\n  const {val, idx} = heap.pop();\n  res.push(val);\n  if (next) heap.push(next);\n}",
          },
          problems: [
            "Merge K Sorted Lists",
            "Kth Smallest Element in a Sorted Matrix",
            "Find K Pairs with Smallest Sums",
            "Smallest Range Covering Elements from K Lists",
          ],
        },
        {
          id: 15,
          phase: "p4",
          name: "0/1 Knapsack",
          desc: "Items can be selected (1) or not (0).",
          strategy: "2D DP array. dp[i][c] = max(exclude, include).",
          anchor: {
            logic:
              "for (let i = 1; i < n; i++) {\n  for (let c = 1; c <= cap; c++) {\n    if (w[i] <= c) \n      dp[i][c] = Math.max(dp[i-1][c], p[i] + dp[i-1][c-w[i]]);\n    else dp[i][c] = dp[i-1][c];\n  }\n}",
          },
          problems: [
            "Partition Equal Subset Sum",
            "Target Sum",
            "Last Stone Weight II",
            "Ones and Zeroes",
          ],
        },
        {
          id: 16,
          phase: "p4",
          name: "Unbounded Knapsack",
          desc: "Items can be selected unlimited times.",
          strategy: "1D DP. If include item, stay at same index.",
          anchor: {
            logic:
              "for (let c = 1; c <= cap; c++) {\n  for (let i = 0; i < n; i++) {\n    if (w[i] <= c) \n      dp[c] = Math.max(dp[c], p[i] + dp[c - w[i]]);\n  }\n}",
          },
          problems: [
            "Coin Change",
            "Rod Cutting",
            "Integer Break",
            "Perfect Squares",
          ],
        },
        {
          id: 17,
          phase: "p4",
          name: "Fibonacci Sequence",
          desc: "Current state depends on previous states.",
          strategy: "dp[i] = dp[i-1] + dp[i-2]. Use two variables.",
          anchor: {
            logic:
              "let n1 = 0, n2 = 1;\nfor (let i = 2; i <= n; i++) {\n  let temp = n1 + n2;\n  n1 = n2;\n  n2 = temp;\n}",
          },
          problems: [
            "Fibonacci Number",
            "Climbing Stairs",
            "House Robber",
            "Jump Game",
          ],
        },
        {
          id: 18,
          phase: "p4",
          name: "Palindromic Subseq",
          desc: "Palindromes within strings.",
          strategy:
            "Check ends. If match, 2+inner. Else max of skipping left or right.",
          anchor: {
            logic:
              "if (s[i] === s[j]) dp[i][j] = 2 + dp[i+1][j-1];\nelse dp[i][j] = Math.max(dp[i+1][j], dp[i][j-1]);",
          },
          problems: [
            "Longest Palindromic Substring",
            "Longest Palindromic Subsequence",
            "Palindromic Substrings",
            "Count Different Palindromic Subsequences",
          ],
        },
        {
          id: 19,
          phase: "p4",
          name: "Longest Common Subsequence",
          desc: "Longest subsequence in two sequences.",
          strategy: "2D DP. Match: diag+1. No match: max(top, left).",
          anchor: {
            logic:
              "if (s1[i] === s2[j]) dp[i][j] = 1 + dp[i-1][j-1];\nelse dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);",
          },
          problems: [
            "Longest Common Subsequence",
            "Edit Distance",
            "Distinct Subsequences",
            "Shortest Common Supersequence",
          ],
        },
        {
          id: 20,
          phase: "p5",
          name: "Topological Sort",
          desc: "Scheduling tasks with dependencies (DAGs).",
          strategy: "Kahn's Algo (BFS with in-degrees).",
          anchor: {
            logic:
              "const q = [];\n// push 0 in-degree nodes\nwhile (q.length) {\n  const u = q.shift();\n  list.push(u);\n  graph[u].forEach(v => {\n    inDegree[v]--;\n    if (inDegree[v] === 0) q.push(v);\n  });\n}",
          },
          problems: [
            "Course Schedule",
            "Alien Dictionary",
            "Sequence Reconstruction",
            "Parallel Courses",
          ],
        },
        {
          id: 21,
          phase: "p5",
          name: "Union Find",
          desc: "Disjoint Set Union (DSU). Connectivity.",
          strategy: "Find with path compression. Union by rank.",
          anchor: {
            logic:
              "find(x) {\n  if (p[x] !== x) p[x] = find(p[x]);\n  return p[x];\n}\nunion(x, y) {\n  p[find(x)] = find(y);\n}",
          },
          problems: [
            "Number of Provinces",
            "Redundant Connection",
            "Accounts Merge",
            "Number of Islands II",
          ],
        },
        {
          id: 22,
          phase: "p5",
          name: "Trie",
          desc: "Prefix Tree. String search/autocomplete.",
          strategy: "Tree where edges are chars.",
          anchor: {
            logic:
              "let node = root;\nfor (const c of word) {\n  if (!node.child[c]) node.child[c] = new Node();\n  node = node.child[c];\n}\nnode.isWord = true;",
          },
          problems: [
            "Implement Trie (Prefix Tree)",
            "Design Add and Search Words Data Structure",
            "Word Search II",
            "Replace Words",
          ],
        },
        {
          id: 23,
          phase: "p5",
          name: "Monotonic Stack",
          desc: "Next Greater/Smaller Element.",
          strategy: "Keep stack sorted. Pop on violation.",
          anchor: {
            logic:
              "for (const n of nums) {\n  while (stack.length && n > stack.peek()) {\n    const popped = stack.pop();\n    // Logic\n  }\n  stack.push(n);\n}",
          },
          problems: [
            "Next Greater Element I",
            "Daily Temperatures",
            "Largest Rectangle in Histogram",
            "Trapping Rain Water",
          ],
        },
        {
          id: 24,
          phase: "p5",
          name: "Matrix Traversal",
          desc: "DFS/BFS on 2D grids.",
          strategy: "Traverse neighbors. Mark visited.",
          anchor: {
            logic:
              "function dfs(r, c) {\n  if (outOfBounds || visited[r][c]) return;\n  visited[r][c] = true;\n  dfs(r+1, c); dfs(r-1, c);\n  dfs(r, c+1); dfs(r, c-1);\n}",
          },
          problems: [
            "Number of Islands",
            "Max Area of Island",
            "Flood Fill",
            "Rotting Oranges",
          ],
        },
        {
          id: 25,
          phase: "p5",
          name: "Backtracking",
          desc: "Brute force. Explore all paths.",
          strategy: "Place, Recurse, Backtrack (Undo).",
          anchor: {
            logic:
              "function backtrack(path) {\n  if (goal) { res.push(path); return; }\n  for (let c of choices) {\n    path.push(c);\n    backtrack(path);\n    path.pop();\n  }\n}",
          },
          problems: [
            "N-Queens",
            "Sudoku Solver",
            "Word Search",
            "Generate Parentheses",
          ],
        },
      ];

      // FULL PROBLEMS DATABASE (One solution per pattern for demo)
      const problemsDB = {
        "Maximum Sum Subarray of Size K": {
          difficulty: "Easy",
          desc: "Find max sum of contiguous subarray of size k.",
          examples: [{ input: "[2,1,5,1,3,2], k=3", output: "9" }],
          constraints: ["k > 0"],
          starterCode: "function solve(k, nums) {\n  return 0;\n}",
          solution:
            "function solve(k, nums) {\n  let maxS=0, wSum=0, start=0;\n  for(let end=0; end<nums.length; end++) {\n    wSum+=nums[end];\n    if(end >= k-1) {\n      maxS = Math.max(maxS, wSum);\n      wSum -= nums[start++];\n    }\n  }\n  return maxS;\n}",
          interviewQs: [{ q: "Negative numbers?", a: "Init max to -Infinity" }],
        },

        "Pair with Target Sum": {
          difficulty: "Easy",
          desc: "Find indices of two numbers that add up to target.",
          examples: [{ input: "[1,2,3,4,6], t=6", output: "[1,3]" }],
          constraints: ["Sorted"],
          starterCode: "function solve(nums, t) {\n  return [-1,-1];\n}",
          solution:
            "function solve(nums, t) {\n  let l=0, r=nums.length-1;\n  while(l<r){\n    let s=nums[l]+nums[r];\n    if(s===t) return [l,r];\n    if(s<t) l++; else r--;\n  }\n  return [-1,-1];\n}",
          interviewQs: [{ q: "Unsorted?", a: "HashMap O(N)" }],
        },

        "Linked List Cycle": {
          difficulty: "Easy",
          desc: "Detect if list has cycle.",
          examples: [{ input: "head=[3,2,0,-4]", output: "true" }],
          constraints: [],
          starterCode: "function hasCycle(head) {\n  return false;\n}",
          solution:
            "function hasCycle(head) {\n  let s=head, f=head;\n  while(f && f.next) {\n    s=s.next; f=f.next.next;\n    if(s===f) return true;\n  }\n  return false;\n}",
          interviewQs: [{ q: "O(1) space?", a: "Yes, Floyd's Algo" }],
        },

        "Merge Intervals": {
          difficulty: "Medium",
          desc: "Merge overlapping intervals.",
          examples: [{ input: "[[1,3],[2,6]]", output: "[[1,6]]" }],
          constraints: [],
          starterCode: "function merge(intervals) {\n  return [];\n}",
          solution:
            "function merge(intervals) {\n  intervals.sort((a,b)=>a[0]-b[0]);\n  const res=[intervals[0]];\n  for(let i=1; i<intervals.length; i++) {\n    let prev=res[res.length-1], curr=intervals[i];\n    if(curr[0]<=prev[1]) prev[1]=Math.max(prev[1], curr[1]);\n    else res.push(curr);\n  }\n  return res;\n}",
          interviewQs: [{ q: "Complexity?", a: "O(N log N)" }],
        },

        "Missing Number": {
          difficulty: "Easy",
          desc: "Find missing number in 0..n",
          examples: [{ input: "[3,0,1]", output: "2" }],
          constraints: [],
          starterCode: "function missing(nums) {\n  return 0;\n}",
          solution:
            "function missing(nums) {\n  let i=0, n=nums.length;\n  while(i<n){\n    let j=nums[i];\n    if(j<n && nums[i]!==nums[j]) [nums[i],nums[j]]=[nums[j],nums[i]];\n    else i++;\n  }\n  for(i=0;i<n;i++) if(nums[i]!==i) return i;\n  return n;\n}",
          interviewQs: [{ q: "XOR approach?", a: "XOR index vs value" }],
        },

        "Reverse Linked List": {
          difficulty: "Easy",
          desc: "Reverse a singly linked list.",
          examples: [{ input: "[1,2,3]", output: "[3,2,1]" }],
          constraints: [],
          starterCode: "function reverse(head) {\n  return null;\n}",
          solution:
            "function reverse(head) {\n  let prev=null, curr=head;\n  while(curr){\n    let next=curr.next;\n    curr.next=prev;\n    prev=curr;\n    curr=next;\n  }\n  return prev;\n}",
          interviewQs: [{ q: "Recursive?", a: "Yes" }],
        },

        "Binary Tree Level Order Traversal": {
          difficulty: "Medium",
          desc: "Level-by-level traversal.",
          examples: [
            { input: "[3,9,20,null,null,15,7]", output: "[[3],[9,20],[15,7]]" },
          ],
          constraints: [],
          starterCode: "function levelOrder(root) {\n  return [];\n}",
          solution:
            "function levelOrder(root) {\n  if(!root) return [];\n  let q=[root], res=[];\n  while(q.length){\n    let len=q.length, row=[];\n    for(let i=0;i<len;i++){\n      let n=q.shift();\n      row.push(n.val);\n      if(n.left) q.push(n.left);\n      if(n.right) q.push(n.right);\n    }\n    res.push(row);\n  }\n  return res;\n}",
          interviewQs: [{ q: "DFS vs BFS?", a: "BFS uses Queue" }],
        },

        "Path Sum": {
          difficulty: "Easy",
          desc: "Has root-to-leaf path summing to target?",
          examples: [{ input: "root, 22", output: "true" }],
          constraints: [],
          starterCode: "function hasPathSum(root, t) {\n  return false;\n}",
          solution:
            "function hasPathSum(root, t) {\n  if(!root) return false;\n  if(!root.left && !root.right && t===root.val) return true;\n  return hasPathSum(root.left, t-root.val) || hasPathSum(root.right, t-root.val);\n}",
          interviewQs: [{ q: "Iterative?", a: "Use Stack" }],
        },

        "Find Median from Data Stream": {
          difficulty: "Hard",
          desc: "Find median of stream.",
          examples: [{ input: "add(1), add(2), med()", output: "1.5" }],
          constraints: [],
          starterCode: "class MedianFinder {}",
          solution:
            "// Requires PriorityQueue implementation in JS\n// 1. MaxHeap (small half)\n// 2. MinHeap (large half)\n// Balance sizes.",
          interviewQs: [{ q: "Why Heaps?", a: "O(1) access" }],
        },

        Subsets: {
          difficulty: "Medium",
          desc: "Find all subsets.",
          examples: [{ input: "[1,2,3]", output: "[[],[1],...]" }],
          constraints: [],
          starterCode: "function subsets(nums) {\n  return [];\n}",
          solution:
            "function subsets(nums) {\n  let res=[[]];\n  for(let n of nums){\n    let len=res.length;\n    for(let i=0;i<len;i++) res.push([...res[i], n]);\n  }\n  return res;\n}",
          interviewQs: [{ q: "Count?", a: "2^N" }],
        },

        "Binary Search": {
          difficulty: "Easy",
          desc: "Find target index.",
          examples: [{ input: "[-1,0,3,5,9], 9", output: "4" }],
          constraints: ["Sorted"],
          starterCode: "function search(nums, t) {\n  return -1;\n}",
          solution:
            "function search(nums, t) {\n  let l=0, r=nums.length-1;\n  while(l<=r){\n    let m=Math.floor((l+r)/2);\n    if(nums[m]===t) return m;\n    if(nums[m]<t) l=m+1; else r=m-1;\n  }\n  return -1;\n}",
          interviewQs: [{ q: "Overflow?", a: "l+(r-l)/2" }],
        },

        "Single Number": {
          difficulty: "Easy",
          desc: "Find unique number.",
          examples: [{ input: "[2,2,1]", output: "1" }],
          constraints: [],
          starterCode: "function singleNumber(nums) {\n  return 0;\n}",
          solution:
            "function singleNumber(nums) {\n  let res=0;\n  for(let n of nums) res^=n;\n  return res;\n}",
          interviewQs: [{ q: "XOR logic?", a: "x^x=0" }],
        },

        "Kth Largest Element in an Array": {
          difficulty: "Medium",
          desc: "Find Kth largest.",
          examples: [{ input: "[3,2,1,5,6,4], k=2", output: "5" }],
          constraints: [],
          starterCode: "function findKth(nums, k) {\n  return 0;\n}",
          solution:
            "function findKth(nums, k) {\n  nums.sort((a,b)=>b-a);\n  return nums[k-1];\n}",
          interviewQs: [{ q: "Faster?", a: "QuickSelect O(N)" }],
        },

        "Merge K Sorted Lists": {
          difficulty: "Hard",
          desc: "Merge K lists.",
          examples: [{ input: "[[1,4,5],[1,3,4]]", output: "[1,1,3,4,4,5]" }],
          constraints: [],
          starterCode: "function mergeK(lists) {\n  return null;\n}",
          solution:
            "// Use MinHeap of size K\n// Push heads, pop min, push next",
          interviewQs: [{ q: "Complexity?", a: "N log K" }],
        },

        "Partition Equal Subset Sum": {
          difficulty: "Medium",
          desc: "Can partition into two equal sums?",
          examples: [{ input: "[1,5,11,5]", output: "true" }],
          constraints: [],
          starterCode: "function canPart(nums) {\n  return false;\n}",
          solution:
            "function canPart(nums) {\n  let sum=nums.reduce((a,b)=>a+b,0);\n  if(sum%2!==0) return false;\n  let t=sum/2, dp=new Array(t+1).fill(false);\n  dp[0]=true;\n  for(let n of nums){\n    for(let i=t; i>=n; i--) dp[i]=dp[i]||dp[i-n];\n  }\n  return dp[t];\n}",
          interviewQs: [{ q: "Why backwards?", a: "Reuse array" }],
        },

        "Coin Change": {
          difficulty: "Medium",
          desc: "Fewest coins to make amount.",
          examples: [{ input: "[1,2,5], 11", output: "3" }],
          constraints: [],
          starterCode: "function coinChange(coins, amt) {\n  return -1;\n}",
          solution:
            "function coinChange(coins, amt) {\n  let dp=new Array(amt+1).fill(Infinity);\n  dp[0]=0;\n  for(let c of coins){\n    for(let i=c; i<=amt; i++) dp[i]=Math.min(dp[i], dp[i-c]+1);\n  }\n  return dp[amt]===Infinity?-1:dp[amt];\n}",
          interviewQs: [{ q: "Greedy?", a: "Fails for some cases" }],
        },

        "Fibonacci Number": {
          difficulty: "Easy",
          desc: "F(n).",
          examples: [{ input: "4", output: "3" }],
          constraints: [],
          starterCode: "function fib(n) {\n  return 0;\n}",
          solution:
            "function fib(n) {\n  if(n<2) return n;\n  let a=0, b=1;\n  for(let i=2; i<=n; i++) { let t=a+b; a=b; b=t; }\n  return b;\n}",
          interviewQs: [{ q: "Space?", a: "O(1)" }],
        },

        "Longest Palindromic Substring": {
          difficulty: "Medium",
          desc: "Longest palindrome in s.",
          examples: [{ input: "babad", output: "bab" }],
          constraints: [],
          starterCode: "function longestPal(s) {\n  return '';\n}",
          solution:
            "function longestPal(s) {\n  let max='';\n  const expand=(l,r)=>{\n    while(l>=0 && r<s.length && s[l]===s[r]) {l--;r++;}\n    if(r-l-1>max.length) max=s.substring(l+1,r);\n  };\n  for(let i=0; i<s.length; i++) { expand(i,i); expand(i,i+1); }\n  return max;\n}",
          interviewQs: [{ q: "Manacher?", a: "O(N)" }],
        },

        "Longest Common Subsequence": {
          difficulty: "Medium",
          desc: "LCS of two strings.",
          examples: [{ input: "abcde, ace", output: "3" }],
          constraints: [],
          starterCode: "function lcs(t1, t2) {\n  return 0;\n}",
          solution:
            "function lcs(t1, t2) {\n  let dp=Array(t1.length+1).fill(0).map(()=>Array(t2.length+1).fill(0));\n  for(let i=1; i<=t1.length; i++){\n    for(let j=1; j<=t2.length; j++){\n      if(t1[i-1]===t2[j-1]) dp[i][j]=1+dp[i-1][j-1];\n      else dp[i][j]=Math.max(dp[i-1][j], dp[i][j-1]);\n    }\n  }\n  return dp[t1.length][t2.length];\n}",
          interviewQs: [{ q: "Space opt?", a: "2 rows" }],
        },

        "Course Schedule": {
          difficulty: "Medium",
          desc: "Can finish courses?",
          examples: [{ input: "2, [[1,0]]", output: "true" }],
          constraints: [],
          starterCode: "function canFinish(n, p) {\n  return true;\n}",
          solution:
            "function canFinish(n, p) {\n  let g=Array.from({length:n},()=>[]), inD=Array(n).fill(0);\n  for(let [c, pre] of p) { g[pre].push(c); inD[c]++; }\n  let q=[], count=0;\n  for(let i=0;i<n;i++) if(inD[i]===0) q.push(i);\n  while(q.length){\n    let u=q.shift(); count++;\n    for(let v of g[u]) { inD[v]--; if(inD[v]===0) q.push(v); }\n  }\n  return count===n;\n}",
          interviewQs: [{ q: "Algorithm?", a: "Kahn's / Topo Sort" }],
        },

        "Number of Provinces": {
          difficulty: "Medium",
          desc: "Connected components.",
          examples: [{ input: "[[1,1,0]...]", output: "2" }],
          constraints: [],
          starterCode: "function findCircleNum(M) {\n  return 0;\n}",
          solution:
            "function findCircleNum(M) {\n  let p=M.map((_,i)=>i), count=M.length;\n  const find=x=>p[x]===x?x:p[x]=find(p[x]);\n  for(let i=0; i<M.length; i++){\n    for(let j=i+1; j<M.length; j++){\n      if(M[i][j] && find(i)!==find(j)) { p[find(i)]=find(j); count--; }\n    }\n  }\n  return count;\n}",
          interviewQs: [{ q: "Time?", a: "N^2 approx" }],
        },

        "Implement Trie (Prefix Tree)": {
          difficulty: "Medium",
          desc: "Insert, Search, StartsWith.",
          examples: [{ input: "insert(apple)", output: "null" }],
          constraints: [],
          starterCode: "class Trie {}",
          solution:
            "class TrieNode{constructor(){this.c={};this.end=false}}\nclass Trie{\n  constructor(){this.root=new TrieNode()}\n  insert(w){let n=this.root;for(let c of w){if(!n.c[c])n.c[c]=new TrieNode();n=n.c[c]}n.end=true}\n}",
          interviewQs: [{ q: "Complexity?", a: "O(L)" }],
        },

        "Next Greater Element I": {
          difficulty: "Easy",
          desc: "Next greater in nums2.",
          examples: [{ input: "[4,1,2], [1,3,4,2]", output: "[-1,3,-1]" }],
          constraints: [],
          starterCode: "function nextGreater(n1, n2) {\n  return [];\n}",
          solution:
            "function nextGreater(n1, n2) {\n  let m=new Map(), s=[];\n  for(let n of n2){\n    while(s.length && n>s[s.length-1]) m.set(s.pop(), n);\n    s.push(n);\n  }\n  return n1.map(n=>m.get(n)||-1);\n}",
          interviewQs: [{ q: "Why stack?", a: "Decreasing monotonic" }],
        },

        "Number of Islands": {
          difficulty: "Medium",
          desc: "Count islands '1'.",
          examples: [{ input: "grid", output: "1" }],
          constraints: [],
          starterCode: "function numIslands(grid) {\n  return 0;\n}",
          solution:
            "function numIslands(grid) {\n  let c=0;\n  const dfs=(i,j)=>{\n    if(i<0||j<0||i>=grid.length||j>=grid[0].length||grid[i][j]==='0')return;\n    grid[i][j]='0';\n    dfs(i+1,j);dfs(i-1,j);dfs(i,j+1);dfs(i,j-1);\n  };\n  for(let i=0;i<grid.length;i++)\n    for(let j=0;j<grid[0].length;j++)\n      if(grid[i][j]==='1'){c++;dfs(i,j);}\n  return c;\n}",
          interviewQs: [{ q: "BFS vs DFS?", a: "Both work" }],
        },

        "N-Queens": {
          difficulty: "Hard",
          desc: "Place N queens.",
          examples: [{ input: "n=4", output: "..." }],
          constraints: [],
          starterCode: "function solveNQueens(n) {\n  return [];\n}",
          solution:
            "// Backtracking with sets for cols, diag1, diag2\n// Recursive function place(row)",
          interviewQs: [{ q: "Time?", a: "N!" }],
        },
      };
