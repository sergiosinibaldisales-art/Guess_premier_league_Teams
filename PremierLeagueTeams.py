import random

premier_league = ['Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 'Burnley', 'Brighton', 'Chelsea', 'Crystal Palace', 'Everton', 'Fulham', 'Leeds United', 'Liverpool', 'Manchester City', 'Manchester United', 'Newcastle United', 'Nottingham Forest', 'Sunderland', 'Tottenham Hotspur', 'West Ham United', 'Wolverhampton Wanderers']

word = random.choice(premier_league).lower()

guessedWord = ["_"] * len(word)
attempts = 10

while attempts > 0:
    # Mostra a palavra atual com espaços entre os caracteres adivinhados
    print('\n Current word: ' + ' '.join(guessedWord))

    # Pede uma letra ao usuário
    guess = input('Guess a letter: ').lower()

    # Verifica se a letra está na palavra
    if guess in word:
        for i in range(len(word)):
            if word[i] == guess:
                guessedWord[i] = guess
        print("Great guess!")
    else:
        attempts -= 1
        print('Wrong guess! Attempts left: ' + str(attempts))

    # Verifica se a palavra foi totalmente adivinhada
    if '_' not in guessedWord:
        print('\n Congratulations! You guessed the team: ' + word)
        break

# Se o loop terminou porque as tentativas acabaram e ainda há
# espaços em branco, informa ao jogador a palavra correta.
if attempts == 0 and '_' in guessedWord:
    print('\nYou\'ve run out of attempts! The team was: ' + word)
