from pgmpy.models import BayesianNetwork
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
        [0, 0.1, 0.3, 0.5, 1, 0, 0.4, 0.7, 0.8, 1],
        [1, 0.9, 0.7, 0.5, 0, 1, 0.6, 0.3, 0.2, 0]
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

print(model.get_independencies())

model.add_cpds(player1card, player2card, player1bet, player2bet, player1response)
model.check_model()