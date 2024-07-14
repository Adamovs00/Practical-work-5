class Node:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class BinaryTree:
    def __init__(self):
        self.root = None

    def insert(self, node, key):
        # If the tree is empty, return a new node
        if node is None:
            return Node(key)
        # Otherwise, recur down the tree
        if key < node.key:
            node.left = self.insert(node.left, key)
        elif key > node.key:
            node.right = self.insert(node.right, key)
        # return the (unchanged) node pointer
        return node

    def inorder(self, root):
        if root:
            self.inorder(root.left)
            print(root.key, end=" ")
            self.inorder(root.right)

    def search(self,root, key):
        # Base Cases: root is null or key is present at root
        if root is None or root.key == key:
            return root

        # Key is greater than root's key
        if root.key < key:
            return self.search(root.right, key)

        # Key is smaller than root's key
        return self.search(root.left, key)

    def deleteNode(self, root, key):
        # Base case
        if root is None:
            return root

        # If the key to be deleted is smaller than the root's key, then it lies in the left subtree
        if key < root.key:
            root.left = self.deleteNode(root.left, key)
        # If the key to be deleted is greater than the root's key, then it lies in the right subtree
        elif key > root.key:
            root.right = self.deleteNode(root.right, key)
        # If key is same as root's key, then this is the node to be deleted
        else:
            # Node with only one child or no child
            if root.left is None:
                return root.right
            elif root.right is None:
                return root.left

            # Node with two children: Get the inorder successor (smallest in the right subtree)
            root.key = self.minValue(root.right)

            # Delete the inorder successor
            root.right = self.deleteNode(root.right, root.key)

        return root

    def minValue(self, root):
        minv = root.key
        while root.left:
            minv = root.left.key
            root = root.left
        return minv

if __name__ == "__main__":
    tree = BinaryTree()
    print('Case a: [49, 38, 65, 97, 60, 76, 13, 27, 5, 1]')
    a = [49, 38, 65, 97, 60, 76, 13, 27, 5, 1]
    #Test insert
    tree.root=tree.insert(tree.root,a[0])
    for i in range(1, len(a)):
        tree.insert(tree.root,a[i])
    print("Tree for case a:")
    tree.inorder(tree.root)
    #Test search
    i=int(input('\nEnter element value to search in the tree: '))
    if tree.search(tree.root, i) is None:
        print(i, "not found")
    else:
        print(i, "found")
    #Test delete
    i = int(input('\nEnter element value to delete in the tree: '))
    print("\nTree before:")
    tree.inorder(tree.root)
    print("\nTree after delete:")
    tree.root=tree.deleteNode(tree.root,i)
    tree.inorder(tree.root)
    print('\nCase b: [149, 38, 65, 197, 60, 176, 13, 217, 5, 11]')
    b = [149, 38, 65, 197, 60, 176, 13, 217, 5, 11]
    for i in range(0, len(b)):
        tree.insert(tree.root,b[i])
    print("Updated tree for case b:")
    tree.inorder(tree.root)
    print('\nCase c: [49, 38, 65, 97, 64, 76, 13, 77, 5, 1, 55, 50, 24]')
    c=[49, 38, 65, 97, 64, 76, 13, 77, 5, 1, 55, 50, 24]
    for i in range(0, len(c)):
        tree.insert(tree.root,c[i])
    print("Updated tree for case c:")
    tree.inorder(tree.root)

