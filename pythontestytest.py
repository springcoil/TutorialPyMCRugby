import theano.tensor as T
import pymc3 as pm
#hyperpriors



model = pm.Model()
with pm.Model() as model:
    home3 = pm.Normal('home', 0, .0001)
    tau_att3 = pm.Gamma('tau_att', .1, .1)
    tau_def3 = pm.Gamma('tau_def', .1, .1)
    intercept3 = pm.Normal('intercept', 0, .0001)
    #team-specific parameters
    atts_star = pm.Normal("atts_star", 
                        mu=0, 
                        tau=tau_att3, 
                        size=num_teams, 
                        value=att_starting_points.values)
    defs_star = pm.Normal("defs_star", 
                        mu=0, 
                        tau=tau_def3, 
                        size=num_teams, 
                        value=def_starting_points.values) 

    atts = pm.Deterministic('regression', atts_star3.copy() - T.mean(atts_star3))
    home_theta3 = pm.Deterministic('regression', T.exp(intercept3 + atts[away_team] + defs[home_team]))
    
    atts = pm.Deterministic('regression', atts_star3.copy() - T.mean(atts_star3))
    home_theta3 = pm.Deterministic('regression', T.exp(intercept3 + atts[away_team] + defs[home_team]))
    # Unknown model parameters
    home_points3 = pm.Poisson('home_points', mu=home_theta3, observed=observed_home_goals)
    away_points3 = pm.Poisson('away_points', mu=home_theta3, observed=observed_away_goals)
    

    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Y)