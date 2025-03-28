import tkinter as tk
from tkinter import Label, Button, Scale, HORIZONTAL, Checkbutton, IntVar, StringVar, OptionMenu

# Creare fereastră principală
root = tk.Tk()
root.title("Calculator de Risc Oncologic")
root.geometry("500x600")

# Variabile pentru factori de risc
sex_var = StringVar(value="Masculin")
varsta_var = tk.IntVar(value=30)
ist_familial_var = IntVar()
varsta_diagnostic_var = IntVar()
fumat_var = IntVar()
alcool_var = IntVar()
dieta_var = IntVar()
activitate_var = IntVar()
expunere_var = IntVar()
boli_var = {"HPV": IntVar(), "Hepatita B/C": IntVar(), "Boli intestinale": IntVar(), "Diabet": IntVar(), "Sindroame genetice": IntVar()}

# Funcție pentru calculul riscului
def calculeaza_risc():
    scor = 0
    if ist_familial_var.get():
        scor += 20
        if varsta_diagnostic_var.get():
            scor += 10
    scor += fumat_var.get() * 10
    scor += alcool_var.get() * 5
    scor += dieta_var.get() * 5
    scor += (1 - activitate_var.get()) * 5
    scor += expunere_var.get() * 10
    scor += sum(var.get() for var in boli_var.values()) * 10
    rezultat_label.config(text=f"📊 Scor de risc: {scor}%")

# Selectare sex
Label(root, text="Sex:").pack()
sex_dropdown = OptionMenu(root, sex_var, "Masculin", "Feminin")
sex_dropdown.pack()

# Selectare vârstă
Label(root, text="Vârsta:").pack()
Scale(root, from_=18, to=90, orient=HORIZONTAL, variable=varsta_var).pack()

# Istoric familial
Checkbutton(root, text="Ai rude de gradul 1 cu cancer?", variable=ist_familial_var).pack()
Checkbutton(root, text="Au fost diagnosticate înainte de 50 ani?", variable=varsta_diagnostic_var).pack()

# Factori de risc
Label(root, text="Factori de risc:").pack()
Checkbutton(root, text="Fumat", variable=fumat_var).pack()
Checkbutton(root, text="Alcool", variable=alcool_var).pack()
Checkbutton(root, text="Dietă nesănătoasă", variable=dieta_var).pack()
Checkbutton(root, text="Lipsa activității fizice", variable=activitate_var).pack()
Checkbutton(root, text="Expunere la substanțe periculoase", variable=expunere_var).pack()

# Boli cronice
Label(root, text="Ai boli cronice?").pack()
for boala, var in boli_var.items():
    Checkbutton(root, text=boala, variable=var).pack()

# Buton de calcul
Button(root, text="Calculează Riscul", command=calculeaza_risc).pack()

# Afișare rezultat
rezultat_label = Label(root, text="📊 Scor de risc: --", font=("Arial", 12))
rezultat_label.pack()

# Pornire aplicație
root.mainloop()
