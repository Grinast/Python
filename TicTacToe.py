# -*- coding: utf-8 -*-
#Tic-tac-toe (or noughts and crosses) is a simple strategy game in which two 
#players take turns placing a mark on a 3x3 board, attempting to make a row, 
#column, or diagonal of three with their mark. In this homework, we will use 
#the tools we've covered in the past two weeks to create a tic-tac-toe simulator
#and evaluate basic winning strategies.

import numpy as np
import random
import time
import matplotlib.pyplot as plt

def create_board():
    return np.zeros((3,3))

def place(board, player, position):
    if board[position] == 0:
        board[position] = player
    return board

#check which cells are empty in current board    
def possibilities(board):
    tuples = np.argwhere(board == 0)
    return tuples

#currect player puts a mark in a rondom available position
def random_place(board, player):
    board[tuple(random.choice(possibilities(board)))] = player
    return board

#win conditions
def row_win(board, player):
    for row in range(3):
        if  (board[row,0] == player) and (board[row,1] == player) and (board[row,2] == player):
            return True
    return False

def col_win(board, player):
    for col in range(3):
        if  (board[0, col] == player) and (board[1,col] == player) and (board[2,col] == player):
            return True
    return False

def diag_win(board, player):
    if  (board[0, 0] == player) and (board[1,1] == player) and (board[2,2] == player):
        return True
    elif  (board[0, 2] == player) and (board[1,1] == player) and (board[2,0] == player):        
        return True
    else: 
        return False
    
def evaluate(board):
    winner = 0
    for player in [1, 2]:
        # Check if `row_win`, `col_win`, or `diag_win` apply. 
    	  # If so, store `player` as `winner`.
          if row_win(board, player) or col_win(board, player) or diag_win(board, player):
              winner = player
    if np.all(board != 0) and winner == 0:
        winner = -1
    return winner

    
def play_game():
    board, winner = create_board(), 0
    player = 1
    while (winner == 0):
        random_place(board, player)
        winner = evaluate(board)
        player = 3 - player #3-1 = 2, 3-2=1
    return winner

#Player 1 chooses middle. Every other move random
def play_strategic_game():
    board, winner = create_board(), 0
    board[1,1] = 1
    player = 2
    while winner == 0:
        random_place(board, player)
        winner = evaluate(board)
        player = 3 - player #3-1 = 2, 3-2=1
    return winner
        
#print(play_game())
    
start = time.time()
record = []
for i in range(1000):
    record.append(play_game())
stop = time.time()

#Compare number of wins. 
plt.hist(record, bins = [-1.2, -0.8, 0.8, 1.2, 1.8, 2.2], normed = "True")
plt.show()
print("Execution time =", stop - start)
print("Player 1 has the advantage when both players play randomly")


start = time.time()
record2 = []
for i in range(1000):
    record2.append(play_strategic_game())
stop = time.time()

#Compare number of wins. 
plt.hist(record2, bins = [-1.2, -0.8, 0.8, 1.2, 1.8, 2.2], normed = "True")
plt.show()
print("Execution time =", stop - start)
print("Player 1 has an even bigger advantage when it chooses the middle cell as the starting move and both players play randomly afterwards.")


"""
Created on Sat Apr 14 09:54:20 2018

@author: Cristina
"""

