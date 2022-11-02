from pgmpy.models import BayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

model = BayesianNetwork([('P1C', 'P2C'), ('P1C', 'P1B'), ('P1B', 'P2B'), ('P2C', 'P2B'),('P1C', 'P1R') ,('P2B', 'P1R')])

player1card = TabularCPD('P1C', 5, [[0.2],[0.2],[0.2],[0.2],[0.2]])
player2card = TabularCPD(
    'P2C', 5, 
    [
        [0, 0.25, 0.25, 0.25, 0.25],
        [0.25, 0, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0, 0.25],
        [0.25, 0.25, 0.25, 0.25, 0]
    ],
    evidence = ['P1C'],
    evidence_card = [5]
)

player1bet = TabularCPD(
    'P1B', 2, 
    [
        [1 ,0.8 ,0.5 ,0.3 ,0],
        [0 ,0.2 ,0.5 ,0.7 ,1]
    ],
    evidence = ['P1C'],
    evidence_card = [5]
)
player2bet = TabularCPD(
    'P2B', 2,
    [
        [1, 0.9, 0.7, 0.5, 0, 1, 0.6, 0.3, 0.2, 0],
        [0, 0.1, 0.3, 0.5, 1, 0, 0.4, 0.7, 0.8, 1]
        
    ],
    evidence = ['P1B', 'P2C'],
    evidence_card = [2, 5]

)
player1response = TabularCPD(
    'P1R', 2,
    [
        [0, 0.1, 0.3, 0.5, 1, 0, 0.4, 0.7, 0.8, 1],
        [1, 0.9, 0.7, 0.5, 0, 1, 0.6, 0.3, 0.2, 0]
    ],
    evidence = ['P2B', 'P1C'],
    evidence_card = [2, 5]
)

model.add_cpds(player1card, player2card, player1bet, player2bet, player1response)
model.check_model()

# player bets = 0, player waits = 1

infer = VariableElimination(model)
a = infer.query(["P1B"], evidence={"P1C": 1})
print(a)

b = infer.query(["P2B"], evidence={"P1B": 0, "P2C": 4})
print(b)


# Schimbam valorile pentru fiecare caz in parte
player1bet = TabularCPD(
    'P1B', 2, 
    [
        [0.8 ,0.6 ,0.4 ,0.1 ,0],
        [0.2 ,0.4 ,0.6 ,0.9 ,1]
    ],
    evidence = ['P1C'],
    evidence_card = [5]
)

player2bet = TabularCPD(
    'P2B', 2,
    [
        [0.8 ,0.6 ,0.4 ,0.1 ,0.05, 0.9, 0.5, 0.3, 0.1, 0],
        [0.2 ,0.4 ,0.6 ,0.9 ,0.95, 0.1, 0.5, 0.7, 0.9, 1]
        
    ],
    evidence = ['P1B', 'P2C'],
    evidence_card = [2, 5]

)

model.add_cpds(player1card, player2card, player1bet, player2bet, player1response)
model.check_model()


# 2
infer = VariableElimination(model)
a = infer.query(["P1B"], evidence={"P1C": 1})
print(a)

b = infer.query(["P2B"], evidence={"P1B": 0, "P2C": 4})
print(b)



player1card1 = TabularCPD('P1C1', 5, [[0.2],[0.2],[0.2],[0.2],[0.2]])
player2card1 = TabularCPD(
    'P2C1', 5, 
    [
        [0, 0.25, 0.25, 0.25, 0.25],
        [0.25, 0, 0.25, 0.25, 0.25],
        [0.25, 0.25, 0, 0.25, 0.25],
        [0.25, 0.25, 0.25, 0, 0.25],
        [0.25, 0.25, 0.25, 0.25, 0]
    ],
    evidence = ['P1C1'],
    evidence_card = [5]
)


# 3
player1card2 = TabularCPD(
    'P2C', 5, 
    [
        #  P1C1 = 0 P1C2 = 0       ////   P1C1 = 0 P1C2 = 1    //// P1C1 = 0   P1C2 = 2    //// P1C1 = 0 P1C2 = 3      ////   P1C1 = 0 P1C2 = 4     
        [0, 0, 0, 0, 0,            0, 0, 0.33, 0.33, 0.33,     0, 0.33, 0, 0.33, 0.33,     0, 0.33, 0.33, 0, 0.33,     0, 0.33, 0.33, 0.33, 0],
        #  P1C1 = 1 P1C2 = 0       ////   P1C1 = 1 P1C2 = 1    //// P1C1 = 1   P1C2 = 2    //// P1C1 = 1 P1C2 = 3      ////   P1C1 = 1 P1C2 = 4  
        [0, 0, 0.33, 0.33, 0.33,   0, 0, 0, 0, 0,              0.33, 0, 0, 0.33, 0.33,     0.33, 0, 0.33, 0, 0.33,     0.33, 0, 0.33, 0.33, 0],
        #  P1C1 = 2 P1C2 = 0       ////   P1C1 = 2 P1C2 = 1    //// P1C1 = 2   P1C2 = 2    //// P1C1 = 2 P1C2 = 3      ////   P1C1 = 2 P1C2 = 4  
        [0, 0.33, 0, 0.33, 0.33,   0.33, 0, 0, 0.33, 0.33,     0, 0, 0, 0, 0,              0.33, 0.33, 0, 0, 0.33,     0.33, 0.33, 0, 0.33, 0],
        #  P1C1 = 3 P1C2 = 0       ////   P1C1 = 3 P1C2 = 1    //// P1C1 = 3   P1C2 = 2    //// P1C1 = 3 P1C2 = 3      ////   P1C1 = 3 P1C2 = 4  
        [0, 0.33, 0.33, 0, 0.33,   0.33, 0, 0.33, 0, 0.33,     0.33, 0, 0, 0.33, 0.33,     0, 0, 0, 0, 0,              0.33, 0.33, 0.33, 0, 0],
        #  P1C1 = 4 P1C2 = 0       ////   P1C1 = 4 P1C2 = 1    //// P1C1 = 4   P1C2 = 2    //// P1C1 = 4 P1C2 = 3      ////   P1C1 = 4 P1C2 = 4  
        [0, 0.33, 0.33, 0.33, 0,   0.33, 0, 0.33, 0.33, 0,     0.33, 0.33, 0, 0.33, 0,     0.33, 0.33, 0.33, 0, 0,     0, 0, 0, 0, 0]
    ],
    evidence = ['P1C1', 'P2C1'],
    evidence_card = [5,5]
)