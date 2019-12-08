
import sqlite3

conn = sqlite3.connect('work007db.sqlite')
cur = conn.cursor()

cur.executescript('''\

Drop Table If Exists History;
Drop Table If Exists Train;

Create Table History(
    id          Integer Primary Key Unique,
    epi         Integer,
    t           Integer,
    y           Real,
    u           Real,
    train_id    Integer
    );

Create Table Train(
    id          Integer Primary Key Unique,
    gamma       Real,
    Nhidden     Real,
    lr_actor    Real,
    lr_critic   Real
    );

''')

conn.commit()
conn.close()
